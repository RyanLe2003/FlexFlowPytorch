import math

TOL = 1.0001e-4

print("---------------- CORRECTNESS CHECK ----------------")
with open("check/result_parallel.txt") as file1, open("check/result_nonparallel.txt") as file2:
    content1 = file1.read().strip().split()
    content2 = file2.read().strip().split()

    if len(content1) != len(content2):
        print("dif length")
        print("Outputs differ.")
    else:
        for idx, (v1, v2) in enumerate(zip(content1, content2)):
            try:
                num1 = float(v1.rstrip(','))
                num2 = float(v2.rstrip(','))
                if not math.isclose(num1, num2, rel_tol=TOL, abs_tol=TOL):
                    print(f"Outputs differ at index {idx}: {num1} vs {num2}")
                    break
            except ValueError:
                if v1 != v2:
                    print(f"V.E. Outputs differ at index {idx}: '{v1}' vs '{v2}'")
                    break
        else:
            print("Outputs match within tolerance!")
