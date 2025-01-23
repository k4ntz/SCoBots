def play(state):
    if state[159] <= -1.020186:
        if state[117] <= 0.168482:
            return 3
        else:
            if state[80] <= -0.084122:
                return 3
            else:
                return 3
    else:
        if state[48] <= -0.133765:
            if state[40] <= 0.397579:
                if state[82] <= 0.102868:
                    if state[74] <= -0.145193:
                        if state[80] <= -0.167315:
                            return 2
                        else:
                            if state[80] <= -0.084559:
                                return 0
                            else:
                                return 3
                    else:
                        if state[79] <= 0.204413:
                            if state[72] <= 0.353517:
                                return 0
                            else:
                                return 3
                        else:
                            if state[27] <= 0.786113:
                                return 2
                            else:
                                return 2
                else:
                    if state[99] <= 0.033314:
                        return 3
                    else:
                        return 3
            else:
                if state[72] <= 0.051892:
                    if state[161] <= 0.911083:
                        return 2
                    else:
                        return 1
                else:
                    return 3
        else:
            return 3
    return -1  # default return -1