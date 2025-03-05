import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import os
from smplx.lbs import batch_rodrigues
from smplx.body_models import SMPLXLayer, SMPLLayer
import trimesh
import pyrender
import imageio
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import plotly.graph_objects as go


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    def init():
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(class_type)

    fig = plt.figure()
    init()
    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
        plt.scatter(data[plot_joints, 0], data[plot_joints, 1], color='b', marker='h', s=15)

    for idx1, idx2 in pose_tree:
        plt.plot([data[idx1, 0], data[idx2, 0]],
                [data[idx1, 1], data[idx2, 1]], color='r', linewidth=2.0)

    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


# def draw_pose_from_cords(img_mat_size, pose_2d, kinematic_tree, radius=2):
#     img = np.zeros(shape=img_mat_size + (3,), dtype=np.uint8)
#     lw = 2
#     pose = pose_2d.astype(np.int32)
#     for i, (idx1, idx2) in enumerate(kinematic_tree):
#         cv2.line(img, (pose[idx1, 0], pose[idx1, 1]), (pose[idx2, 0], pose[idx2, 1]), (255, 255, 255), lw)
#
#     for i, uv in enumerate(pose_2d):
#         point = tuple(uv.astype(np.int32))
#         cv2.circle(img, point, radius, COLORS[i % len(COLORS)], -1)
#     return img

def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    figure = plt.figure()
    # ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), projection='3d')
    ax = Axes3D(figure)
#     ax.set_ylim(-1, 1)
#     ax.set_xlim(-1, 1)
#     ax.set_zlim(-1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=110, azim=90)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    for chain, color in zip(kinematic_tree, colors):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2.0, color=color)
#     ax.set_aspect(1)
# #     plt.axis('off')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     plt.savefig(savePath)
    plt.show()

def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset=None, title=None):
#     matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(0, 800)
        ax.set_xlim(0, 800)
        ax.set_zlim(0, 5000)
        # ax.set_ylim(-0.75, 0.75)
        # ax.set_xlim(-0.75, 0.75)
        # ax.set_zlim(-0.75, 0.75)
        if title is not None:
            ax.set_title(title)

    motion = motion.reshape(motion.shape[0], -1, 3)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue','red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.scatter(motion[index, :, 0], motion[index, :, 1], motion[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=2.0, color=color)
#         ax.set_aspect('equal')
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


# radius = 10*offsets
def plot_3d_motion_kit(save_path, kinematic_tree, joints, title, figsize=(5, 5), interval=100, radius=246 * 12):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'magenta', 'black', 'green', 'blue', 'red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=2.0, color=color)
        #         print(trajec[:index, 0].shape)
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_motion_gt_pred(save_path, kinematic_tree, gt_joints, pred_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        for ax in axs:
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            fig.suptitle(title, fontsize=20)
            ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz, ax):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    def update(index):
        for i, ax in enumerate(axs):
            ax.lines = []
            ax.collections = []
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            MINS = motions_min[i]
            MAXS = motions_max[i]
            trajec = motions_traj[i]
            data = motions_data[i]

            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                        MAXS[2] - trajec[index, 1], ax)

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                        color='blue')

            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                        color=color)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    # (seq_len, joints_num, 3)
    
    motions_data = []
    motions_traj = []
    motions_min = []
    motions_max = []
    colors = ['red', 'blue', 'black', 'red', 'blue',
                'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    for i, joints in enumerate((gt_joints, pred_joints)):
        data = joints.copy().reshape(len(joints), -1, 3)
        frame_number = data.shape[0]

        MINS = data.min(axis=0).min(axis=0)
        motions_min.append(MINS)
        MAXS = data.max(axis=0).max(axis=0)
        motions_max.append(MAXS)
        height_offset = MINS[1]

        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        motions_traj.append(trajec)

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        motions_data.append(data)

    axs = []
    fig = plt.figure(figsize=(20,10))
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    init()

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        # ax.lines = []
        # ax.collections = []

        # del ax.lines[:]  
        # del ax.collections[:]

        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_plotly(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    """
    Строит 3D-анимацию движения с помощью Plotly и сохраняет её в HTML-файл.
    
    Исходно координаты считаются в формате (x, y, z), где y – вертикальная ось.
    Для корректного отображения в Plotly (по умолчанию вертикальная ось – z)
    происходит преобразование: (x, y, z) → (x, z, y).
    
    Параметры:
      save_path: путь для сохранения HTML-файла;
      kinematic_tree: список цепочек (каждая цепочка – список индексов суставов);
      joints: массив суставов, который можно привести к форме (num_frames, joints, 3);
      title: заголовок графика;
      figsize: размер фигуры (будет умножен на 100 для пикселей);
      fps: частота кадров;
      radius: диапазон осей (используется при задании лимитов сцены).
    """
    # Форматирование заголовка (с HTML-разрывами строк)
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '<br>'.join([' '.join(title_sp[:10]),
                             ' '.join(title_sp[10:20]),
                             ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '<br>'.join([' '.join(title_sp[:10]),
                             ' '.join(title_sp[10:])])
    
    # Приводим данные к форме (num_frames, joints, 3)
    data = joints.copy().reshape(len(joints), -1, 3)  # исходная система: (x, y, z)
    # print(data.shape[1])
    num_frames = data.shape[0]
    
    # Вычисляем глобальные минимумы и максимумы (по исходным координатам)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    # Поднимаем модель: сдвигаем ось y так, чтобы самый низкий сустов был на уровне 0
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset

    # Сохраняем исходную (нецентрованную) траекторию корневого сустава
    # (будем использовать для отрисовки траектории движения)
    trajec = data[:, 0, :].copy()  # (x, y, z) для корневого сустава

    # Центрируем скелет по горизонтальным осям (x и z)
    data[:, :, 0] -= data[:, 0:1, 0]
    data[:, :, 2] -= data[:, 0:1, 2]
    
    # Преобразуем координаты для Plotly: (x, y, z) → (new_x, new_y, new_z)
    # где:
    #   new_x = исходный x (центрованный),
    #   new_y = исходный z (центрованный),
    #   new_z = исходный y (с поправкой по высоте).
    data_new = np.empty_like(data)
    data_new[..., 0] = data[..., 0]       # new_x
    data_new[..., 1] = data[..., 2]       # new_y
    data_new[..., 2] = data[..., 1]       # new_z

    # Аналогичное преобразование для траектории (корневой сустава):
    trajec_new = np.empty_like(trajec)
    trajec_new[..., 0] = trajec[..., 0]        # x
    trajec_new[..., 1] = trajec[..., 2]        # y (из исходного z)
    trajec_new[..., 2] = trajec[..., 1]          # z (из исходного y)

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    # Подготавливаем кадры анимации
    frames = []
    for i in range(num_frames):
        frame_data = []
        # Отрисовка плоскости (земли) на уровне new_z = 0.
        # Вычисляем пределы по x и y:
        minx = MINS[0] - trajec[i, 0]
        maxx = MAXS[0] - trajec[i, 0]
        # Для y используем исходные значения z
        miny = MINS[2] - trajec[i, 2]
        maxy = MAXS[2] - trajec[i, 2]
        plane_x = [minx, minx, maxx, maxx]
        plane_y = [miny, maxy, maxy, miny]
        plane_z = [0, 0, 0, 0]
        plane_trace = go.Mesh3d(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='gray',
            opacity=0.5,
            showscale=False,
            name='Plane',
            hoverinfo='skip'
        )
        frame_data.append(plane_trace)
        
        # Траектория движения корневого сустава, отрисованная на земле (new_z = 0)
        if i > 0:
            traj_x = (trajec_new[:i, 0] - trajec_new[i, 0]).tolist()
            traj_y = (trajec_new[:i, 1] - trajec_new[i, 1]).tolist()
            traj_z = [0] * i
        else:
            traj_x, traj_y, traj_z = [], [], []
        traj_trace = go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=traj_z,
            mode='lines',
            line=dict(color='blue', width=1),
            name='Trajectory'
        )
        frame_data.append(traj_trace)
        
        # Отрисовка каждой кинематической цепи
        for j, chain in enumerate(kinematic_tree):
            chain_x = data_new[i, chain, 0].tolist()
            chain_y = data_new[i, chain, 1].tolist()
            chain_z = data_new[i, chain, 2].tolist()
            lw = 4 if j < 5 else 2
            chain_trace = go.Scatter3d(
                x=chain_x,
                y=chain_y,
                z=chain_z,
                mode='lines',
                line=dict(color=colors[j], width=lw),
                name=f'Chain {j}',
                showlegend=False
            )
            frame_data.append(chain_trace)
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Инициализируем фигуру с данными первого кадра
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Обновляем параметры сцены:
    # new_x и new_y – горизонтальная плоскость, new_z – вертикальная
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-radius/2, radius/2], autorange=False, title="X"),
            yaxis=dict(range=[-radius/2, radius/2], autorange=False, title="Y"),
            zaxis=dict(range=[0, radius], autorange=False, title="Z"),
            camera=dict(
                # Камеру можно настроить по необходимости
                eye=dict(x=0, y=6.5, z=3.75)
            )
        ),
        width=figsize[0]*100,
        height=figsize[1]*100,
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000/fps, "redraw": True},
                                    "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    sliders = [{
        "steps": [
            {
                "args": [[str(k)], {"frame": {"duration": 1000/fps, "redraw": True},
                                     "mode": "immediate"}],
                "label": str(k),
                "method": "animate"
            } for k in range(num_frames)
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "y": 0,
        "currentvalue": {"font": {"size": 12}, "prefix": "Frame: ", "visible": True, "xanchor": "center"},
        "len": 0.9
    }]
    fig.update_layout(sliders=sliders)    
    fig.write_html(save_path)


def animate_smplx_from_joints(model_path, joints, 
                            out_file="smplx_animation.gif", 
                            fps=20, device='cpu',
                            cam_height=3.0, 
                            cam_distance=4.0):
    """
    Создаёт анимацию SMPL-X модели с естественными движениями
    
    Параметры:
        model_path: путь к SMPL-X модели
        joints: массив суставов в формате (num_frames, 22, 3)
        out_file: выходной файл анимации
        fps: частота кадров
        device: вычислительное устройство (cpu/cuda)
        cam_height: высота камеры над моделью
        cam_distance: дистанция от камеры до модели
    """
    # Инициализация модели
    device = torch.device(device)
    model = SMPLXLayer(model_path, gender='neutral').to(device)
    
    # Конфигурация суставов SMPL-X
    SMPLX_JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 
        'left_knee', 'right_knee', 'spine2', 
        'left_ankle', 'right_ankle', 'spine3', 
        'left_foot', 'right_foot', 'neck', 
        'left_collar', 'right_collar', 'head',
        'left_shoulder', 'right_shoulder', 
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist'
    ]
    
    # Иерархия суставов для SMPL-X
    SMPLX_KINEMATIC_CHAIN = [
        [0, 3, 6, 9, 12, 15],  # Центральная ось (таз -> голова)
        [0, 1, 4, 7, 10],      # Левая нога
        [0, 2, 5, 8, 11],      # Правая нога
        [12, 13, 16, 18, 20],  # Левая рука
        [12, 14, 17, 19, 21]   # Правая рука
    ]

    # Ограничения суставов (в радианах)
    JOINT_CONSTRAINTS = {
        'left_elbow': (-np.pi/2, 0),
        'right_elbow': (-np.pi/2, 0),
        'left_knee': (0, np.pi*0.8),
        'right_knee': (0, np.pi*0.8),
        'left_shoulder': (-np.pi/3, np.pi/3),
        'right_shoulder': (-np.pi/3, np.pi/3)
    }

    # Подготовка данных
    joints = torch.tensor(joints, device=device, dtype=torch.float32)
    num_frames = joints.shape[0]
    
    # Получение T-позы
    with torch.no_grad():
        body_pose = torch.zeros(1, 63, device=device)
        body_pose_rotmat = batch_rodrigues(body_pose.view(-1, 3)).view(1, 21, 3, 3)
        rest_pose = model(body_pose=body_pose_rotmat).joints.squeeze()

    # Инициализация рендерера
    renderer = pyrender.OffscreenRenderer(1024, 768)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, -cam_height, cam_distance]
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

    frames = []
    for i in range(num_frames):
        current_joints = joints[i]
        
        # 1. Вычисление корневой ориентации
        root_orient = compute_root_orientation(current_joints, rest_pose)
        
        # 2. Иерархическое вычисление вращений
        body_pose = torch.zeros(21, 3, device=device)
        parent_rotmats = {0: torch.eye(3, device=device)}
        
        for chain in SMPLX_KINEMATIC_CHAIN:
            for j in range(1, len(chain)):
                child_idx = chain[j]
                parent_idx = chain[j-1]
                joint_name = SMPLX_JOINT_NAMES[child_idx]
                
                # Вычисление относительного вращения
                rot = compute_joint_rotation(
                    current_joints[[parent_idx, child_idx]],
                    rest_pose[[parent_idx, child_idx]],
                    parent_rotmats[parent_idx]
                )
                
                # Применение ограничений
                if joint_name in JOINT_CONSTRAINTS:
                    rot = apply_joint_constraint(rot, *JOINT_CONSTRAINTS[joint_name])
                
                # Сохранение вращения
                body_pose[child_idx-1] = rot
                parent_rotmats[child_idx] = parent_rotmats[parent_idx] @ rotation_matrix(rot)

        # 3. Формирование параметров модели
        body_pose_rotmat = batch_rodrigues(body_pose.view(-1, 3)).view(1, 21, 3, 3)
        
        output = model(
            global_orient=batch_rodrigues(root_orient.view(1, 3)),
            body_pose=body_pose_rotmat,
            transl=current_joints[0].view(1, 3)
        )

        # 4. Рендеринг
        mesh = trimesh.Trimesh(
            output.vertices.detach().cpu().numpy().squeeze(),
            model.faces,
            process=False
        )
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1,0,0]))
        
        scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.96, 1.0])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        
        frames.append(renderer.render(scene)[0])

    imageio.mimsave(out_file, frames, fps=fps)
    renderer.delete()
    print(f"Анимация сохранена: {os.path.abspath(out_file)}")


def compute_root_orientation(current_joints, rest_pose):
    """Вычисляет ориентацию корпуса с учётом позвоночника"""
    # Вектор от таза к шее в текущей позе
    spine_vector = current_joints[12] - current_joints[0]
    
    # Вектор от таза к шее в T-позе
    rest_spine = rest_pose[12] - rest_pose[0]
    
    # Нормализация векторов
    rest_spine_normalized = F.normalize(rest_spine, dim=0)
    spine_vector_normalized = F.normalize(spine_vector, dim=0)
    
    # Вычисление вращения между векторами
    axis = torch.cross(rest_spine_normalized, spine_vector_normalized, dim=0)
    cos_angle = torch.dot(rest_spine_normalized, spine_vector_normalized)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # Фильтрация малых вращений
    if angle < 1e-6:
        return torch.zeros(3, device=current_joints.device)
    
    return axis * angle

def compute_joint_rotation(current_bones, rest_bones, parent_rotmat):
    """Вычисляет вращение сустава относительно родителя"""
    current_dir = F.normalize(current_bones[1] - current_bones[0], dim=0)
    rest_dir = F.normalize(rest_bones[1] - rest_bones[0], dim=0)
    
    # Учёт вращения родителя
    rest_dir_global = parent_rotmat @ rest_dir
    
    # Вычисление необходимого вращения
    axis = torch.cross(rest_dir_global, current_dir, dim=0)
    cos_angle = torch.dot(rest_dir_global, current_dir)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # Нормализация
    angle = torch.clamp(angle, -np.pi, np.pi)
    return axis * angle if angle > 1e-6 else torch.zeros_like(axis)


def apply_joint_constraint(rotation, min_angle, max_angle):
    """Применяет ограничения к вращению сустава"""
    angle = torch.norm(rotation)
    if angle == 0:
        return rotation
    
    clamped_angle = torch.clamp(angle, min_angle, max_angle)
    return rotation * (clamped_angle / angle)

def rotation_matrix(axis_angle):
    """Создаёт матрицу вращения из axis-angle представления"""
    angle = torch.norm(axis_angle)
    if angle < 1e-6:
        return torch.eye(3, device=axis_angle.device)
    
    axis = axis_angle / angle
    return batch_rodrigues(axis.unsqueeze(0)).squeeze(0)

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Вычисляет матрицу поворота, которая поворачивает вектор vec1 в направление vec2.
    Если векторы вырожденные или почти совпадают, возвращается единичная матрица.
    """
    a = vec1 / (np.linalg.norm(vec1) + 1e-8)
    b = vec2 / (np.linalg.norm(vec2) + 1e-8)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3, dtype=np.float32)
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + kmat + np.matmul(kmat, kmat) * ((1 - c) / (s**2))
    return R

def render_smplx_animation(joints, mesh_path, output_path="animation.mp4", 
                           kinematic_tree=None, device='cpu', fps: int = 30,
                           mapping=None):
    """
    Рендерит анимацию SMPL-X модели на основе последовательности joints, 
    применяя инверсную кинематику для маппинга joints (актуальных для HumanML3D)
    в параметры модели SMPL-X.
    
    Аргументы:
        mesh_path (str): Путь к файлу меша SMPL-X модели (.npz).
        joints (numpy.ndarray): Массив формы (sequence_length, num_joints, 3),
                                где num_joints актуальны для HumanML3D.
        output_path (str, optional): Путь для сохранения анимации (например, MP4).
        kinematic_tree (list[list[int]] или dict, optional): 
            Кинематическое дерево для HumanML3D. Если передается список списков,
            то оно преобразуется в dict: {child: parent, ...}.
        device (str, optional): Устройство вычислений ('cpu' или 'cuda').
        fps (int, optional): Количество кадров в секунду.
        mapping (dict, optional): Словарь соответствия индексов между HumanML3D и SMPL-X.
            Например, для SMPL-X body_pose (21 сустав) можно задать:
            {
                1: human_joint_index_1,
                2: human_joint_index_2,
                ...
                21: human_joint_index_21
            }
            Если не задан, предполагается, что индексы совпадают (1->1, 2->2, ...).
    """
    device = torch.device(device)

    # Преобразуем kinematic_tree, если он передан как список списков
    if kinematic_tree is not None and isinstance(kinematic_tree, list):
        kin_dict = {}
        for chain in kinematic_tree:
            if not chain:
                continue
            root = chain[0]
            if root not in kin_dict:
                kin_dict[root] = -1
            for i in range(1, len(chain)):
                child = chain[i]
                parent = chain[i-1]
                kin_dict[child] = parent
        kinematic_tree = kin_dict

    # Если mapping не задан, предполагаем прямое соответствие: SMPL-X индекс == HumanML3D индекс
    if mapping is None:
        # SMPL-X body_pose ожидает 21 матрицу для суставов 1..21
        mapping = {i: i for i in range(1, 22)}

    # Загрузка SMPL-X модели (путь указывает на директорию с файлами модели)
    smplx_model = SMPLXLayer(model_path=os.path.dirname(mesh_path),
                             model_type='smplx',
                             gender='neutral').to(device)

    sequence_length, num_human_joints, _ = joints.shape

    # Предполагаем, что первый кадр является rest-позой (эталонное положение)
    rest_joints = joints[0]

    frames = []
    
    # Инициализация сцены с камерой и светом
    scene = pyrender.Scene()

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    scene.add_node(camera_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    light_node = pyrender.Node(light=light, matrix=light_pose)
    scene.add_node(light_node)

    for t in range(sequence_length):
        current_joints = joints[t]  # (num_human_joints, 3)
        
        # Глобальная ориентация (для корневого сустава, предполагаем, что индекс 0 – корень)
        default_global_dir = rest_joints[1] - rest_joints[0]
        actual_global_dir = current_joints[1] - current_joints[0]
        R_global = rotation_matrix_from_vectors(default_global_dir, actual_global_dir)
        global_orient = torch.tensor(R_global, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Вычисление локальных поворотов для суставов SMPL-X
        # Итоговое body_pose должно быть [1, 21*9] (для 21 сустава)
        body_pose_matrices = []
        for smplx_idx in range(1, 22):  # SMPL-X суставы с 1 по 21
            # Определяем соответствующий индекс в HumanML3D
            human_idx = mapping.get(smplx_idx, None)
            if human_idx is None:
                # Если соответствие отсутствует, используем единичную матрицу
                R_local = np.eye(3, dtype=np.float32)
            else:
                # Определяем родителя в HumanML3D для данного сустава
                parent_human = kinematic_tree.get(human_idx, -1) if kinematic_tree is not None else (human_idx - 1)
                if parent_human < 0:
                    R_local = np.eye(3, dtype=np.float32)
                else:
                    default_vec = rest_joints[human_idx] - rest_joints[parent_human]
                    current_vec = current_joints[human_idx] - current_joints[parent_human]
                    R_local = rotation_matrix_from_vectors(default_vec, current_vec)
                    
                    # !!! Здесь можно добавить коррекцию twist, если есть дополнительная информация
            body_pose_matrices.append(R_local)
        
        # Собираем 21 матрицу в тензор [1, 21, 3, 3] -> затем в [1,189]
        body_pose_tensor = torch.tensor(np.stack(body_pose_matrices, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
        body_pose_flat = body_pose_tensor.view(1, -1)
        
        betas = torch.zeros([1, 10], dtype=torch.float32).to(device)
        
        # Вычисляем вершины модели SMPL-X
        smplx_output = smplx_model(body_pose=body_pose_flat, global_orient=global_orient, betas=betas, return_verts=True)
        vertices = smplx_output.vertices.detach().cpu().numpy()[0]
        faces = smplx_model.faces

        mesh = trimesh.Trimesh(vertices, faces)
        mesh_render = pyrender.Mesh.from_trimesh(mesh)

        # Удаляем предыдущие меши
        for node in list(scene.mesh_nodes):
            scene.remove_node(node)
            
        scene.add_node(pyrender.Node(mesh=mesh_render))

        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, depth = r.render(scene)
        frames.append(color)
        r.delete()

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Анимация сохранена в: {output_path}")


def plot_3d_motion_old(motion, pose_tree, class_type, save_path, interval=300, excluded_joints=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)
        ax.set_title(class_type)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        ax.lines = []
        ax.collections = []
        if excluded_joints is None:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='b', marker='h', s=15)
        else:
            plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
            ax.scatter(data[index, plot_joints, 0], data[index, plot_joints, 1], data[index, plot_joints, 2], color='b', marker='h', s=15)

        for idx1, idx2 in pose_tree:
            ax.plot([data[index, idx1, 0], data[index, idx2, 0]],
                    [data[index, idx1, 1], data[index, idx2, 1]], [data[index, idx1, 2], data[index, idx2, 2]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


def plot_2d_motion(motion, pose_tree, axis_0, axis_1, class_type, save_path, interval=300):
    matplotlib.use('Agg')

    fig = plt.figure()
    plt.title(class_type)
    # ax = fig.add_subplot(111, projection='3d')
    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        plt.clf()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plt.scatter(data[index, :, axis_0], data[index, :, axis_1], color='b', marker='h', s=15)
        for idx1, idx2 in pose_tree:
            plt.plot([data[index, idx1, axis_0], data[index, idx2, axis_0]],
                    [data[index, idx1, axis_1], data[index, idx2, axis_1]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_multi_motion(motion_list, kinematic_tree, save_path, interval=50, dataset=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    frame_number = motion_list[0].shape[0]
    # dim (frame, joints, xyz)
    # print(data.shape)
    print("Number of motions %d" % (len(motion_list)))
    def update(index):
        ax.lines = []
        ax.collections = []
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for motion in motion_list:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                          linewidth=4.0, color=color)
        plt.axis('off')

#         ax.set_xticks([])
#         ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()



