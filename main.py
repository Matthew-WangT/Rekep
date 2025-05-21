import torch
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from omnigibson.robots.fetch import Fetch
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)
import time

class Main:
    def __init__(self, scene_file, visualize=False):
        # 加载全局配置
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize 
        # 设置随机种子
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # 初始化关键点提议器和约束生成器
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # 初始化环境
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)
        # 设置逆运动学求解器（用于可达性成本）
        assert isinstance(self.env.robot, Fetch), "IK求解器假设机器人是Fetch机器人"
        ik_solver = IKSolver(
            robot_description_path=self.env.robot.robot_arm_descriptor_yamls[self.env.robot.default_arm],
            robot_urdf_path=self.env.robot.urdf_path,
            eef_name=self.env.robot.eef_link_names[self.env.robot.default_arm],
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )
        # 初始化子目标求解器和路径求解器
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # 初始化可视化工具
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        # 重置环境
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        # ====================================
        # = 关键点提议和约束生成
        # ====================================
        if rekep_program_dir is None:
            print(f"points.shape: {points.shape}")
            time_start = time.time()
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            time_end = time.time()
            print(f"time cost: {1000 * (time_end - time_start)} ms")
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = 执行任务
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        # 更新干扰序列
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # 加载元数据
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # 注册需要跟踪的关键点
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        # 加载约束
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # 记录哪些关键点在优化中可以移动
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # 第一个关键点总是末端执行器，因此是可移动的

        # 主循环
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = 决定是否回溯
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # 根据约束决定回溯到哪个阶段
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    if len(path_constraints) == 0:
                        break
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # 应用干扰
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = 获取优化计划
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = 执行
                # ====================================
                # 确定是否进入下一个阶段
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # 如果完成，保存视频并返回
                    if self.stage == self.program_info['num_stages']: 
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # 进入下一个阶段
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        # 获取下一个子目标
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        # 获取下一个路径
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        # if self.visualize:
        #     self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # 对路径进行样条插值
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # 添加夹爪动作
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # 更新阶段
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        assert self.is_grasp_stage + self.is_release_stage <= 1, "不能同时是抓取和释放阶段"
        if self.is_grasp_stage:
            self.env.open_gripper()
        self.action_queue = []
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        # 更新关键点可移动掩码
        for i in range(1, len(self.keypoint_movable_mask)):
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        # 执行抓取动作
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
    
    def _execute_release_action(self):
        # 执行释放动作
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    if args.apply_disturbance:
        assert args.task == 'pen' and args.use_cached_query, 'disturbance sequence is only defined for cached scenario'

    # ====================================
    # = 笔任务干扰序列
    # ====================================
    def stage1_disturbance_seq(env):
        """
        在阶段0中移动笔，当机器人试图抓住笔时
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # 干扰序列
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.08, 0.0, 0.0])
        orn1 = T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/4])), orn0)
        pose1 = np.concatenate([pos1, orn1])
        pos2 = pos1 + np.array([0.10, 0.0, 0.0])
        orn2 = T.quat_multiply(T.euler2quat(np.array([0, 0, -np.pi/2])), orn1)
        pose2 = np.concatenate([pos2, orn2])
        control_points = np.array([pose0, pose1, pose2])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                pen.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage2_disturbance_seq(env):
        """
        在阶段1中从夹爪中取出笔，当机器人试图重新定位笔时
        """
        apply_disturbance = env.is_grasping()
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # 干扰序列
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pose1 = np.array([-0.30, -0.15, 0.71, -0.7071068, 0, 0, 0.7071068])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if apply_disturbance:
                if counter < 20:
                    if counter > 15:
                        env.robot.release_grasp_immediately()
                    else:
                        pass
                elif counter < len(pose_seq) + 20:
                    env.robot.release_grasp_immediately()
                    pose = pose_seq[counter - 20]
                    pos, orn = pose[:3], pose[3:]
                    pen.set_position_orientation(pos, orn)
                    counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage3_disturbance_seq(env):
        """
        在阶段2中移动笔架，当机器人试图将笔放入笔架时
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # 干扰序列
        pos0, orn0 = holder.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.02, -0.15, 0.0])
        orn1 = orn0
        pose1 = np.concatenate([pos1, orn1])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=5)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                holder.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1

    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_red_pen.json',
            'instruction': 'reorient the red pen and drop it upright into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
        'apple': {
            'scene_file': './configs/og_scene_file_red_pen.json',
            'instruction': 'pick up the green apple and drop it into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
        'almond_milk': {
            'scene_file': './configs/og_scene_file_red_pen.json',
            'instruction': 'pick up the box of almond milk and drop it into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
        'knife': {
            'scene_file': './configs/og_scene_file_red_pen.json',
            'instruction': 'pick up the knife and drop it into the black pen holder',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
    }
    task = task_list['almond_milk']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)