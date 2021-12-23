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


tests_run, tests_passed = 0, 0
for root, dir, files in os.walk("./tests"):
    for file in files:
        if file.endswith(".tlp"):
            cleanup()
            file_path = root+"/"+file
            txt_file = file[:-3] + "txt"
            txt_file_path = root+"/"+txt_file

            compare = txt_file in files
            if compare:
                f = open("tmp.txt", "w")
            else:
                print(f"Output File for {file_path} doesn't exist yet.")
                print(f"    Making one...")
                f = open(txt_file_path, "w")

            compile_result = run(
                ["python3", "tulip.py", root+"/"+file],
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
                tests_run += 1
                diff_result = run(
                    ["diff", "tmp.txt", txt_file_path], capture_output=True)
                if diff_result.returncode != 0:
                    print(f"[TEST]: {file_path} -- Failed.")
                    print(f"[DIFF]: {diff_result.stdout.decode('utf-8')}")
                else:
                    print(f"[TEST]: {file_path} -- Passed.")
                    tests_passed += 1

run(["rm", "tmp.txt"], capture_output=True)
print("----------------------------------")
print(f"{tests_passed} of {tests_run} tests passed")
print("----------------------------------")
print("Starting MyPy")
run(['mypy', 'tulip.py'])
