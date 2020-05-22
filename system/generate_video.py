"""Given a folder of pngs, this script generates a mp4 video."""
import os
import shutil
import argparse

from ns_vqa_dart.bullet import util

render_frequency = 5
speed_up_factor = 27 / render_frequency


def main(args: argparse.Namespace):
    # create_video_from_captures_dir(args)

    root_dir = "/home/mguo/outputs/system/demo_z0_v2/0521_1825_head_tilt"
    src_dirs = os.path.join(root_dir, "png")
    out_dir = os.path.join(root_dir, "mp4")
    util.delete_and_create_dir(out_dir)
    for vid in sorted(os.listdir(src_dirs)):
        src_dir = os.path.join(src_dirs, vid)
        dst_path = os.path.join(out_dir, f"{vid}.mp4")
        create_video_per_dir(args, src_dir, dst_path)

    # Creates one video from multiple png directories.
    # create_single_video_for_dirs(args, frame_dirs)

    # if TRIM:
    #     trim_command = f"ffmpeg -ss 00:00:00 -t 00:01:44 -i {args.dst_path}.mp4 -vcodec copy -acodec copy {args.dst_path}_trimmed.mp4"
    #     os.system(trim_command)


def create_video_per_dir(args, src_dir, dst_path):
    paths = [os.path.join(src_dir, p) for p in sorted(os.listdir(src_dir))]
    create_ordered_dir(paths, args.reordered_dir)
    generate_video(args.reordered_dir, dst_path, fps=40, speed_up_factor=40 / 25)
    delete_dir(args.reordered_dir)


def create_single_video_for_dirs(args, frame_dirs):
    paths = collect_frames_from_dirs(frame_dirs)
    print(f"Number of frames: {len(paths)}")
    create_ordered_dir(paths, args.reordered_dir)
    generate_video(
        src_dir=args.reordered_dir,
        dst_path=args.dst_path,
        fps=40,
        speed_up_factor=40 / 25,
    )


def collect_frames_from_dirs(frame_dirs):
    paths = []
    for frame_dir in frame_dirs:
        for f in sorted(os.listdir(frame_dir)):
            paths.append(os.path.join(frame_dir, f))
    return paths


def generate_video(src_dir: str, dst_path: str, fps: int, speed_up_factor=None):
    assert not os.path.exists(dst_path)

    if speed_up_factor is not None:
        arg = f"setpts=PTS/{speed_up_factor},fps={fps}"
    else:
        arg = f"fps={fps}"

    # command = f'ffmpeg -i {src_dir}/%06d.png -filter:v "{arg}" -vb 20M -vcodec mpeg4 -y {dst_path}'
    command = f'ffmpeg -i {src_dir}/%06d.png -frames:v 25000 -filter:v "setpts=PTS/{speed_up_factor},fps={fps}" -vb 20M -vcodec libx264 -pix_fmt yuv420p -preset veryslow -y {dst_path}'
    os.system(command)


def create_ordered_dir(paths, reordered_dir):
    util.delete_and_create_dir(reordered_dir)
    for idx, p in enumerate(paths):
        dst_path = os.path.join(reordered_dir, f"{idx:06}.png")
        shutil.copyfile(p, dst_path)


def delete_dir(reordered_dir):
    # Delete the temporary reordered directory.
    shutil.rmtree(reordered_dir)


def create_video_from_captures_dir(args):
    run_dir = os.path.join(args.run_root_dir, args.run_name)

    for pov in ["first", "third"]:
        print(f"Generating video for pov: {pov}...")
        src_dir = os.path.join(run_dir, "Captures/temp", pov, "rgb")
        # First pov images are not ordered, so we need to re-order them.
        reorder = pov == "first"

        # Create a reordered directory of images that ffmpeg can run on.
        if reorder:
            create_orderered_dir(src_dir, args.reordered_src_dir)
            src_dir = args.reordered_src_dir

        dst_path = os.path.join(run_dir, f"{pov}.mp4")
        generate_video(src_dir=src_dir, dst_path=dst_path, fps=args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "run_name",
    #     type=str,
    #     help="The directory containing the png images to convert into video format.",
    # )
    # parser.add_argument(
    #     "--run_root_dir",
    #     type=str,
    #     default="/home/mguo/outputs/system/t1/0404",
    #     help="The directory containing the png images to convert into video format.",
    # )
    # parser.add_argument(
    #     "--pov",
    #     type=str,
    #     default="first",
    #     help="The point of view to generate video for.",
    # )
    parser.add_argument(
        "--reordered_dir",
        type=str,
        default="/home/mguo/tmp_ffmpeg",
        help="The directory containing the png images to convert into video format.",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default="/home/mguo/test.mp4",
        help="The directory containing the png images to convert into video format.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="The frames per second.",
    )
    args = parser.parse_args()
    main(args=args)
