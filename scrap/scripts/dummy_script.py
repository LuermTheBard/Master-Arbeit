import sys


def calc_2_plus_2():
    print("2 + 2 =", 2 + 2)


if __name__ == "__main__":

    print("Dummy Script is running...")

    if len(sys.argv) == 2:

        if sys.argv[1] == "calc_2_plus_2":
            print("Run command calc_2_plus_2")
            calc_2_plus_2()
        else:
            print("Command unknown")
    else:
        print("No command given")
