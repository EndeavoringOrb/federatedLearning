import subprocess
import time
import sys
import os

while True:
    try:
        # Run the script
        process = subprocess.Popen(
            ["pyEnv/Scripts/python", os.path.abspath("trainClient.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Print output of the script
        for line in process.stdout:
            sys.stdout.write(line)

        # Wait for the script to complete
        process.wait()

    except Exception as e:
        print(f"An error occurred: {e}. Restarting the script...")

    # Wait for a short time before restarting
    time.sleep(5)
