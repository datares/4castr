import argparse
from lib.cli.static_session import Static_Session
from lib.cli.live_session import Live_Session
from stable_baselines import PPO2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='choose between optimization, validation, train.')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Name for the session"')
    parser.add_argument('-b', '--brain', type=str, default=None,
                        help='the model path"')
    parser.add_argument('-s', '--stock', type=str, default=None,
                        help='Stock if fine_tuning"')
    parser.add_argument('-e', '--episodes', type=int, default=500,
                        help='number of episodes to run')
    parser.add_argument('-i', '--initial_invest', type=int, default=1000,
                        help='initial investment amount')

    # Parses arguments
    args = parser.parse_args()
    input_b = args.stock is not None and args.brain is not None

    if args.mode == "optimize":
        s = Static_Session(args.mode,
                           args.episodes,
                           args.initial_invest,
                           args.name,
                           args.brain,
                           False)
        s.run_optimization()

    if args.mode == "train":
        s = Static_Session(args.mode, args.episodes,
                           args.initial_invest,
                           args.name, args.brain, False)
        s.run_train()

    if args.mode == "finetune" and input_b:
        s = Static_Session(args.mode, args.episodes,
                           args.initial_invest, args.name,
                           args.brain, False)
        s.fine_tune(args.stock, args.brain)

    if args.mode == "validation" and args.brain is not None:
        s = Static_Session(args.mode, args.episodes,
                           args.initial_invest, args.name,
                           args.brain, False)
        s.run_test(args.brain)

    if args.mode == "test" and args.brain is not None:
        s = Static_Session(args.mode, args.episodes,
                           args.initial_invest, args.name,
                           args.brain, False)
        model = PPO2.load(args.brain)
        s.run_test(model, False, finetune=True, verbose=True, out_file=True)

    if args.mode == "live" and input_b:
        sesh = Live_Session(args.mode, args.initial_invest,
                            args.name, args.brain,
                            args.stock)
        sesh.go_live()
