import sys

def main():
    count = sys.getrefcount(None)
    print(f"0b{count:b}")

if __name__ == "__main__":
    main()

# 3.11 prints: 0b1000000101111
# 3.12 prints: 0b11111111111111111111111111111111 <-- None Refcount, a read is just a read.
