#pragma once

#include "../include/uint128.cuh"
#include "../include/Utils.cuh"
#include "../include/SecretKey.cuh"
#include "../include/Scheme_23.cuh"
#include "../include/Key.cuh"


class CCMM_Scheme{
public:
    Scheme_23& scheme;
    Context_23& context;
    CCMM_Scheme(Scheme_23& scheme);

};