"""Generates the bash script that executes main_sim_stack_new_w_delay.py with
various arguments for vision."""


def main():
    for pose_type in ["gt", "gt_delay", "vision"]:
        with open(f"{pose_type}.sh", "w") as f:
            for scene in [1, 2, 3]:
                for shape in ["box", "cylinder"]:
                    for size in ["small", "large"]:
                        if pose_type in ["gt_delay", "vision"]:
                            script_name = "main_sim_stack_new_w_delay"
                        elif pose_type in ["gt"]:
                            script_name = "main_sim_stack_new"
                        lines = [
                            f"python {script_name}.py",
                            f"  --pose_path ~/demo_poses/{pose_type}/scene_{scene}/{size}_{shape}.json",
                            f"  --scene {scene}",
                            f"  --shape {shape}",
                            f"  --size {size}",
                        ]
                        if pose_type == "vision":
                            lines.append("  --use_vision")
                        cmd = " \\\n".join(lines)
                        cmd += "\n\n"

                        f.write(cmd)


if __name__ == "__main__":
    main()
