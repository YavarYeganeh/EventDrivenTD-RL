from __future__ import annotations

import argparse
import time

from sim.commands import CMD_INITIALIZE, CMD_RUN_STEP, CMD_WAITING, CMD_STOP
from sim.config import SimulationConfig
from sim.interfaces import EventDrivenSystem
from sim.loader import load_system


def run_evaluation(system: EventDrivenSystem) -> None:
    config = system.config

    if config.warmup_time > 0:
        system.run_until(config.warmup_time)

    system.enable_policy_initialization()

    if config.load_policy:
        system.request_policy_sync()

    system.run_until(system.now() + config.horizon_time)


def initialize_training(system: EventDrivenSystem) -> None:
    config = system.config

    if config.warmup_time > 0:
        system.run_until(config.warmup_time)

    system.enable_policy_initialization()

    if config.load_policy:
        system.request_policy_sync()

    if config.prefill_time > 0:
        system.run_until(system.now() + config.prefill_time)

    system.control.set_command(CMD_WAITING)
    print(
        f"Simulation worker: scenario={config.scenario_id} "
        "initialization completed. Waiting for next command."
    )


def run_training_step(system: EventDrivenSystem) -> None:
    config = system.config
    start_position = system.control.get_position()
    empty_intervals = 0

    while not system.experiences.has_completed_experience(start_position):
        system.run_until(system.now() + config.step_time)
        empty_intervals += 1

        if empty_intervals > config.max_empty_intervals:
            print(
                "Warning: no completed experience after "
                f"{config.max_empty_intervals} simulation intervals."
            )
            break

    system.request_policy_sync()
    system.control.set_command(CMD_WAITING)


def run_training_loop(system: EventDrivenSystem) -> None:
    try:
        while True:
            command = system.control.get_command()

            if command == CMD_INITIALIZE:
                initialize_training(system)
            elif command == CMD_RUN_STEP:
                run_training_step(system)
            elif command == CMD_STOP:
                print(
                    "Controller requested simulation shutdown: "
                    f"scenario={system.config.scenario_id}"
                )
                break

            time.sleep(system.config.command_poll_seconds)
    finally:
        system.experiences.close()
        system.control.close()
        system.close()


def build_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        training_mode=bool(args.train),
        warmup_time=float(args.warmup_time),
        prefill_time=float(args.prefill_time),
        horizon_time=float(args.horizon_time),
        step_time=float(args.sim_interval_time),
        command_poll_seconds=float(args.cmd_wait),
        scenario_id=args.scenario_id,
        random_seed=args.seed,
        load_policy=bool(args.load_agent),
        policy_path=args.load_path or "",
        max_empty_intervals=int(args.max_empty_intervals),
        results_dir=args.train_results_dir or "",
        tensor_dir=args.mm_dir or "",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        default="sim.placeholder_system:build_system",
        help=(
            "Import path to a system builder, for example "
            "'examples.simple_queue.system:build_system'."
        ),
    )

    # Compatibility arguments used by the current train_online.py launcher.
    parser.add_argument("--scenario_id", default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--agent", default="")
    parser.add_argument("--load_agent", action="store_true")
    parser.add_argument("--load_path", default="")
    parser.add_argument("--train_results_dir", default="")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--sim_interval_time", type=float, default=1.0)
    parser.add_argument("--cmd_shm", default="")
    parser.add_argument("--rb_shm", default="")
    parser.add_argument("--rb_cap", type=int, default=0)
    parser.add_argument("--mm_dir", default="")

    # Generic simulation timing.
    parser.add_argument("--warmup_time", type=float, default=0.0)
    parser.add_argument("--prefill_time", type=float, default=0.0)
    parser.add_argument("--horizon_time", type=float, default=1000.0)
    parser.add_argument("--cmd_wait", type=float, default=0.05)
    parser.add_argument("--max_empty_intervals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)

    # Reward-related compatibility arguments. They are intentionally not used
    # by the generic worker; adapters may inspect cli_args if needed.
    parser.add_argument("--r_alpha", default=None)
    parser.add_argument("--r_beta", default=None)
    parser.add_argument("--r_zeta", default=None)
    parser.add_argument("--r_gamma", default=None)
    parser.add_argument("--r_delta", default=None)
    parser.add_argument("--r_phi", default=None)
    parser.add_argument("--no_constraint_term", action="store_true")
    parser.add_argument("--skip_r_systen", action="store_true")

    return parser


def main() -> None:
    parser = make_parser()
    args, unknown_args = parser.parse_known_args()

    config = build_config_from_args(args)
    system = load_system(
        args.system,
        config=config,
        cli_args=args,
        unknown_args=unknown_args,
    )

    if system.config.training_mode:
        run_training_loop(system)
    else:
        run_evaluation(system)


if __name__ == "__main__":
    main()
