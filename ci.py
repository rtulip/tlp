import os
from subprocess import run
from argparse import ArgumentParser


def cleanup(name='output'):
    try:
        os.remove(name)
    except FileNotFoundError:
        pass

    try:
        os.remove(f'{name}.asm')
    except FileNotFoundError:
        pass
    try:
        os.remove(f'{name}.o')
    except FileNotFoundError:
        pass


def run_test(file_path: str, txt_file_path: str, compare: bool, verbose: bool = True, greedy_exit: bool = False) -> int:
    cleanup()
    if compare:
        f = open("tmp.txt", "w")
    else:
        print(f"Output File for {file_path} doesn't exist yet.")
        print(f"    Making one...")
        f = open(txt_file_path, "w")

    compile_result = run(
        ["python3", "tulip.py", file_path],
        capture_output=True
    )

    f.write(f"args: {compile_result.args}\n")
    f.write(f"-------- stdout ----------\n")
    f.write(f"{compile_result.stdout.decode('utf-8')}")
    f.write(f"--------------------------\n")
    f.write(f"-------- stderr ----------\n")
    f.write(f"{compile_result.stderr.decode('utf-8')}")
    f.write(f"--------------------------\n")
    f.write(f"exit code: {compile_result.returncode}\n")

    if compile_result.returncode == 0:
        output_result = run(["./output"], capture_output=True)
        f.write(f"##########################\n")
        f.write(f"args: {output_result.args}\n")
        f.write(f"-------- stdout ----------\n")
        f.write(f"{output_result.stdout.decode('utf-8')}")
        f.write(f"--------------------------\n")
        f.write(f"-------- stderr ----------\n")
        f.write(f"{output_result.stderr.decode('utf-8')}")
        f.write(f"--------------------------\n")
        f.write(f"exit code: {output_result.returncode}\n")

    else:
        output_result = None

    f.close()

    if compare:
        diff_result = run(
            ["diff", "tmp.txt", txt_file_path], capture_output=True)
        if diff_result.returncode != 0:
            print(f"[TEST]: {file_path} -- Failed.")
            print(f"[DIFF]: {diff_result.stdout.decode('utf-8')}")
            if greedy_exit:
                exit(1)
        else:
            if verbose:
                print(f"[TEST]: {file_path} -- Passed.")
            return 1

    return 0


def main():
    parser = ArgumentParser(description='Run Tests.')
    parser.add_argument('--test', type=str)
    parser.add_argument('--update', type=str)
    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--greedy', action='store_true', default=False)

    args = parser.parse_args()
    if args.test:
        file_path = args.test
        txt_file_path = file_path[:-3] + "txt"
        run_test(file_path, txt_file_path, True, args.greedy)
    elif args.update:
        file_path = args.update
        txt_file_path = file_path[:-3] + "txt"
        run_test(file_path, txt_file_path, False, False)
    else:
        tests_run, tests_passed = 0, 0
        for root, dir, files in os.walk("./tests"):
            for file in files:
                if file.endswith(".tlp"):
                    file_path = root+"/"+file
                    txt_file = file[:-3] + "txt"
                    txt_file_path = root+"/"+txt_file
                    compare = txt_file in files
                    if compare:
                        tests_run += 1
                    tests_passed += run_test(file_path, txt_file_path,
                                             compare, verbose=not args.filter, greedy_exit=args.greedy)

        run(["rm", "tmp.txt"], capture_output=True)
        print("----------------------------------")
        print(f"{tests_passed} of {tests_run} tests passed")
        print("----------------------------------")
        print("Starting MyPy")
        run(['mypy', 'tulip.py'])


if __name__ == "__main__":
    main()
