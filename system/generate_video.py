"""Given a folder of pngs, this script generates a mp4 video."""
import os
import shutil
import argparse

from ns_vqa_dart.bullet import util

render_frequency = 5
speed_up_factor = 27 / render_frequency


def main(args: argparse.Namespace):
    src_dir = os.path.join(args.run_dir, "Captures/temp", args.pov, "rgb")

    # First pov images are not ordered, so we need to re-order them.
    reorder = args.pov == "first"

    # Create a reordered directory of images that ffmpeg can run on.
    if reorder:
        create_orderered_dir(src_dir, args.reordered_src_dir)
        src_dir = args.reordered_src_dir

    dst_path = os.path.join(args.run_dir, f"{args.pov}.mp4")
    generate_video(src_dir=src_dir, dst_path=dst_path, fps=args.fps)

    # Delete the temporary reordered directory.
    if reorder:
        shutil.rmtree(args.reordered_src_dir)

    # if TRIM:
    #     trim_command = f"ffmpeg -ss 00:00:00 -t 00:01:44 -i {args.dst_path}.mp4 -vcodec copy -acodec copy {args.dst_path}_trimmed.mp4"
    #     os.system(trim_command)


def generate_video(src_dir: str, dst_path: str, fps: int, speed_up_factor: int = None):
    assert not os.path.exists(dst_path)

    if speed_up_factor is not None:
        arg = f"setpts=PTS/{speed_up_factor},fps={fps}"
    else:
        arg = f"fps={fps}"

    command = f'ffmpeg -i {src_dir}/%06d.png -filter:v "{arg}" -vb 20M -vcodec mpeg4 -y {dst_path}'
    os.system(command)


def create_orderered_dir(orig_dir, reordered_dir):
    util.delete_and_create_dir(reordered_dir)
    for idx, fname in enumerate(sorted(os.listdir(orig_dir))):
        src_path = os.path.join(orig_dir, fname)
        dst_path = os.path.join(reordered_dir, f"{idx:06}.png")
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="/home/mguo/outputs/system/t1/0404/2020_05_15_12_51_33",
        help="The directory containing the png images to convert into video format.",
    )
    parser.add_argument(
        "--pov",
        type=str,
        default="first",
        help="The point of view to generate video for.",
    )
    parser.add_argument(
        "--reordered_src_dir",
        type=str,
        default="/home/mguo/tmp_ffmpeg",
        help="The directory containing the png images to convert into video format.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="The frames per second.",
    )
    args = parser.parse_args()
    main(args=args)
