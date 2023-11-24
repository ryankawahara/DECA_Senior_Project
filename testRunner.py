import subprocess
import os
import pathlib
# Path to a Python interpreter that runs any Python script
# under the virtualenv /path/to/virtualenv/

# python_bin_path = "/mnt/c/Users/ryryk/Documents/Github/SeniorProject/DECA_Senior_Project/.updatedVersVenv/bin/python"

# Convert the Unix-style path to a Windows-readable path
# python_bin = os.path.normpath(python_bin_path)
#
# windows_path = "C:" + python_bin.replace("/", "\\")


# Path to the script that must run under the virtualenv
# script_file = "must/run/under/virtualenv/script.py"

# command = f"bash -c {python_bin} demos/demo_reconstruct_neutral.py -i TestSamples/latestTests --saveDepth True --saveObj True"

# command = f"bash -c '{python_bin} importTest.py'"
# command = f"bash -c '{python_bin} --version'"
# print(os.path.normpath(os.getcwd()))
wsl_path = pathlib.PureWindowsPath(os.getcwd()).as_posix()
wsl_path = wsl_path.replace('C:', '/mnt/c')


print(f"here {wsl_path}")


venv_path = os.path.join(wsl_path, ".copiedVenvs")

bin_path = os.path.join(venv_path, "bin", "python")
bin_path = bin_path.replace("\\", "/")

# path_obj = Path(bin_path)
# unix_path = path_obj.as_posix()
# print(unix_path, os.path.exists(unix_path))

# command = f"bash -c {bin_path} -m pip list"
libPath = os.path.join(os.getcwd(), ".copiedVenvs", "bin")
# print(f"current {os.path.join(os.getcwd())}")

my_env = os.environ.copy()
print(my_env["PATH"])
print("~~~~")
print(f"libpath = {libPath}")
my_env["PATH"] = libPath + ";" + my_env["PATH"]
print(my_env["PATH"])
# venv_path =
command = f"bash -c '{bin_path} demos/demo_reconstruct_neutral.py -i TestSamples/latestTests --saveDepth True --saveObj True'"

# command = f"bash -c '{bin_path} -m pip install ninja'"




# command = f"bash -c '{bin_path} -m pip list'"
# command = f"bash -c '{bin_path} importTest.py'"
# command = f"bash -c '{bin_path} demos/demo_reconstruct_neutral.py -i TestSamples/latestTests --saveDepth True --saveObj True'"

# command = f"source /path/to/venv/bin/activate && {bin_path} --version"
# command = [bin_path, "--version"]
#
# # Run the subprocess and capture its output
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)

# Wait for the subprocess to complete and capture its output
stdout, stderr = process.communicate()
#
# # Print the captured output
print("Standard Output:")
print(stdout.decode('utf-8').strip())  # Decode the bytes to a string and remove leading/trailing spaces
print(stderr.decode('utf-8').strip())  # Decode the bytes to a string and remove leading/trailing spaces

# Run the subprocess and capture its output
# process = subprocess.Popen([python_bin, script_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the subprocess to complete and capture its output
# stdout, stderr = process.communicate()

# # Print the captured output
# print("Standard Output:")
# print(stdout.decode('utf-8'))  # Decode the bytes to a string for printing
#
# print("Standard Error:")
# print(stderr.decode('utf-8'))  # Decode the bytes to a string for printing
