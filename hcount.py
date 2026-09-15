import subprocess, sys, re
out = subprocess.run([sys.executable,'main.py','--max-frames','100','--no-viz'],
                     capture_output=True, text=True)
for line in (out.stdout + out.stderr).splitlines():
    if any(k in line for k in ('H selected','E selected','Valid poses','Invalid')):
        print(line.strip())
