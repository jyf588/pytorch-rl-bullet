"""Given a folder of pngs, this script generates a mp4 video."""
import os
import shutil
import argparse


POV = "first"
TRIM = False


def main(args: argparse.Namespace):
    fps = 30
    render_frequency = 5
    speed_up_factor = 27 / render_frequency

    # First pov images are not ordered, so we need to re-order them.
    src_dir = args.src_dir
    reorder = False
    if POV == "first":
        reorder = True

    if reorder:
        tmp_dir = "/home/mguo/tmp_ffmpeg"
        os.makedirs(tmp_dir)
        for idx, fname in enumerate(sorted(os.listdir(args.src_dir))):
            src_path = os.path.join(args.src_dir, fname)
            dst_path = os.path.join(tmp_dir, f"{idx:06}.png")
            shutil.copyfile(src_path, dst_path)
        src_dir = tmp_dir

    # Demo command
    # demo_command = f'ffmpeg -i {src_dir}/%06d.png -filter:v "setpts=PTS/{speed_up_factor},fps={fps}" -vb 20M -vcodec mpeg4 -y {args.dst_path}.mp4'
    debug_command = f'ffmpeg -i {src_dir}/%06d.png -filter:v "fps={fps}" -vb 20M -vcodec mpeg4 -y {args.dst_path}.mp4'

    os.system(debug_command)
    if reorder:
        shutil.rmtree(tmp_dir)

    if TRIM:
        trim_command = f"ffmpeg -ss 00:00:00 -t 00:01:44 -i {args.dst_path}.mp4 -vcodec copy -acodec copy {args.dst_path}_trimmed.mp4"
        os.system(trim_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        # default="/Users/michelleguo/workspace/lucas/unity/Captures/temp/third/rgb",
        # default="/Users/michelleguo/Desktop/dash/demo_video/vision/0507",
        # default="/home/mguo/unity/Builds/LinuxBuildLocalhost0512/0514_Captures/temp/third/rgb",
        default="/home/mguo/outputs/system/t1/0404/2020_05_14_14_21_40/Captures/temp/first/rgb",
        # default="/home/mguo/outputs/system/t1/0404/2020_05_14_14_21_40/Captures/temp/third/rgb",
        help="The directory containing the png images to convert into video format.",
    )
    # parser.add_argument(
    #     "--temp_path",
    #     type=str,
    #     default="/Users/michelleguo/Desktop/tmp.mp4",
    #     help="The destination path to write the mp4 video to.",
    # )
    parser.add_argument(
        "--dst_path",
        type=str,
        # default="/Users/michelleguo/Desktop/test.mp4",
        default="/home/mguo/test",
        help="The destination path to write the mp4 video to.",
    )
    args = parser.parse_args()
    main(args=args)
