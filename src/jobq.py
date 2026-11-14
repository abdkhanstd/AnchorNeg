"""Run a list of train.py jobs on the local GPUs, one job per GPU at a time.
Usage: python jobq.py jobs.txt   (one argument string per line)"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.environ.get("SCRIPT", "train.py")
PY = os.environ.get("PYEXE", os.path.join(os.path.dirname(os.path.dirname(HERE)), ".venv", "Scripts", "python.exe"))
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1").split(",")]


def main():
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    logdir = os.path.join(os.path.dirname(HERE), "results", "logs")
    os.makedirs(logdir, exist_ok=True)
    running = {}
    while jobs or running:
        for g in GPUS:
            p = running.get(g)
            if p is not None and p[0].poll() is not None:
                print("done", g, p[1], "rc", p[0].returncode, flush=True)
                running.pop(g)
            if g not in running and jobs:
                j = jobs.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONIOENCODING="utf-8",
                           TOKENIZERS_PARALLELISM="false")
                log = open(os.path.join(logdir, re.sub(r"[^A-Za-z0-9_.-]", "", j.replace("--", "_"))[:150] + ".log"), "w")
                running[g] = (subprocess.Popen([PY, os.path.join(HERE, SCRIPT)] + j.split(),
                                               cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), j)
                print("start", g, j, flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
