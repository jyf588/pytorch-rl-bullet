"""Runs the demo in bullet."""

from demo.options import OPTIONS
from demo.env import DemoEnvironment


def main():
    env = DemoEnvironment(
        opt=OPTIONS, observation_mode="gt", visualize_bullet=True,
    )

    while 1:
        done = env.step()

        # We are done.
        if done:
            break


if __name__ == "__main__":
    main()
