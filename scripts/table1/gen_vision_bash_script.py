"""Generates the bash script that executes main_sim_stack_new_w_delay.py with
various arguments for vision."""


def main():
    with open("vision.sh", "w") as f:
        for shape in ["box", "cylinder"]:
            for size in ["small", "large"]:
                lines = [
                    f"python main_sim_stack_new_w_delay.py",
                    f"  --pose_path ~/demo_poses/vision/{size}_{shape}.json",
                    f"  --shape {shape}",
                    f"  --size {size}",
                ]
                cmd = " \\\n".join(lines)
                cmd += "\n\n"

                f.write(cmd)
                # f.write(
                #     f"python main_sim_stack_new_w_delay.py \\"
                #     "--pose_path ~/demo_poses/vision/{size}_{shape}.json"
                # )


if __name__ == "__main__":
    main()
