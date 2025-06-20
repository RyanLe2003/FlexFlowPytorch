print("---------------- CORRECTNESS CHECK ----------------")
with open("logs/result_parallel.txt") as f1, open("logs/result_nonparallel.txt") as f2:
    if f1.read().strip() == f2.read().strip():
        print("Outputs match!")
    else:
        print("Outputs differ.")