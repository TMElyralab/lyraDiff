#pragma once
#include "stdio.h"
#include "stdlib.h"

// be consistent with lyradiff
int8_t float_to_int8_rn_host(float x)
{
    int8_t  res;
    int32_t tmp;
    if (x >= 0) {
        tmp = int(x + 0.5);
        tmp = tmp > 127 ? 127 : tmp;
        res = int8_t(tmp);
    }
    else {
        tmp = int(x - 0.5);
        tmp = tmp < -127 ? -127 : tmp;
        res = int8_t(tmp);
    }
    return res;
}