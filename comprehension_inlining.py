# Pep 709: comprehension inlining

def comp():
    d = {x: x*x for x in range(10)}
    l = [x*x for x in range(10)]
    s = {x*x for x in range(10)}

    v = {x for x in locals()}  # conatins d, l, s
    print(d, l, s, v sep="\n")

