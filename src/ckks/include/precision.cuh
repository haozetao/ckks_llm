#pragma once
#include "uint128.cuh"
#include "Utils.cuh"

struct Stats {
    double Real, Imag, L2;
};

struct PrecisionStats {
    Stats MaxDelta;
    Stats MinDelta;
    Stats MaxPrecision;
    Stats MinPrecision;
    Stats MeanDelta;
    Stats MeanPrecision;
    Stats MedianDelta;
    Stats MedianPrecision;
    double STDFreq;
    double STDTime;

    struct Dist {
        double Prec;
        int Count;
    };

    std::vector<Dist> RealDist, ImagDist, L2Dist;
    int cdfResol;

    std::string String() const {
        char buffer[512];
        snprintf(buffer, sizeof(buffer),
                 "\n _________________________________\n"
                 "|    Log2 | REAL  | IMAG  | L2    |\n"
                 "|MIN Prec | %.2f | %.2f | %.2f |\n"
                 "|MAX Prec | %.2f | %.2f | %.2f |\n"
                 "|AVG Prec | %.2f | %.2f | %.2f |\n"
                 "|MED Prec | %.2f | %.2f | %.2f |\n"
                 "===================================\n",
                 MinPrecision.Real, MinPrecision.Imag, MinPrecision.L2,
                 MaxPrecision.Real, MaxPrecision.Imag, MaxPrecision.L2,
                 MeanPrecision.Real, MeanPrecision.Imag, MeanPrecision.L2,
                 MedianPrecision.Real, MedianPrecision.Imag, MedianPrecision.L2);
        return std::string(buffer);
    }
};

Stats deltaToPrecision(const Stats& c) {
    return {log2(1 / c.Real), log2(1 / c.Imag), log2(1 / c.L2)};
}

void calcCDF(const std::vector<double>& precs, std::vector<PrecisionStats::Dist>& res, int cdfResol) {
    std::vector<double> sortedPrecs = precs;
    std::sort(sortedPrecs.begin(), sortedPrecs.end());
    double minPrec = sortedPrecs[0];
    double maxPrec = sortedPrecs.back();
    for (int i = 0; i < cdfResol; i++) {
        double curPrec = minPrec + double(i) * (maxPrec - minPrec) / double(cdfResol);
        for (size_t countSmaller = 0; countSmaller < sortedPrecs.size(); countSmaller++) {
            if (sortedPrecs[countSmaller] >= curPrec) {
                res[i].Prec = curPrec;
                res[i].Count = countSmaller;
                break;
            }
        }
    }
}

Stats calcMedian(std::vector<Stats>& values) {
    std::vector<double> tmp(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        tmp[i] = values[i].Real;
    }
    std::sort(tmp.begin(), tmp.end());
    for (size_t i = 0; i < values.size(); i++) {
        values[i].Real = tmp[i];
    }

    for (size_t i = 0; i < values.size(); i++) {
        tmp[i] = values[i].Imag;
    }
    std::sort(tmp.begin(), tmp.end());
    for (size_t i = 0; i < values.size(); i++) {
        values[i].Imag = tmp[i];
    }

    for (size_t i = 0; i < values.size(); i++) {
        tmp[i] = values[i].L2;
    }
    std::sort(tmp.begin(), tmp.end());
    for (size_t i = 0; i < values.size(); i++) {
        values[i].L2 = tmp[i];
    }

    size_t index = values.size() / 2;
    if (values.size() % 2 == 1 || index + 1 == values.size()) {
        return {values[index].Real, values[index].Imag, values[index].L2};
    }

    return {(values[index].Real + values[index + 1].Real) / 2,
            (values[index].Imag + values[index + 1].Imag) / 2,
            (values[index].L2 + values[index + 1].L2) / 2};
}

PrecisionStats GetPrecisionStats(const std::vector<cuDoubleComplex>& valuesTest, const std::vector<cuDoubleComplex>& valuesWant) {
    PrecisionStats prec;
    prec.MaxDelta = {0, 0, 0};
    prec.MinDelta = {1, 1, 1};
    prec.MeanDelta = {0, 0, 0};
    prec.cdfResol = 500;

    prec.RealDist.resize(prec.cdfResol);
    prec.ImagDist.resize(prec.cdfResol);
    prec.L2Dist.resize(prec.cdfResol);

    std::vector<double> precReal(valuesWant.size());
    std::vector<double> precImag(valuesWant.size());
    std::vector<double> precL2(valuesWant.size());

    std::vector<Stats> diff(valuesWant.size());

    for (size_t i = 0; i < valuesWant.size(); i++) {
        double deltaReal = std::abs(valuesTest[i].x - valuesWant[i].x);
        double deltaImag = std::abs(valuesTest[i].y - valuesWant[i].y);
        double deltaL2 = std::sqrt(deltaReal * deltaReal + deltaImag * deltaImag);
        precReal[i] = log2(1 / deltaReal);
        precImag[i] = log2(1 / deltaImag);
        precL2[i] = log2(1 / deltaL2);

        diff[i] = {deltaReal, deltaImag, deltaL2};

        prec.MeanDelta.Real += deltaReal;
        prec.MeanDelta.Imag += deltaImag;
        prec.MeanDelta.L2 += deltaL2;

        if (deltaReal > prec.MaxDelta.Real) {
            prec.MaxDelta.Real = deltaReal;
        }
        if (deltaImag > prec.MaxDelta.Imag) {
            prec.MaxDelta.Imag = deltaImag;
        }
        if (deltaL2 > prec.MaxDelta.L2) {
            prec.MaxDelta.L2 = deltaL2;
        }

        if (deltaReal < prec.MinDelta.Real) {
            prec.MinDelta.Real = deltaReal;
        }
        if (deltaImag < prec.MinDelta.Imag) {
            prec.MinDelta.Imag = deltaImag;
        }
        if (deltaL2 < prec.MinDelta.L2) {
            prec.MinDelta.L2 = deltaL2;
        }
    }

    prec.MeanDelta.Real /= valuesWant.size();
    prec.MeanDelta.Imag /= valuesWant.size();
    prec.MeanDelta.L2 /= valuesWant.size();

    calcCDF(precReal, prec.RealDist, prec.cdfResol);
    calcCDF(precImag, prec.ImagDist, prec.cdfResol);
    calcCDF(precL2, prec.L2Dist, prec.cdfResol);

    prec.MinPrecision = deltaToPrecision(prec.MaxDelta);
    prec.MaxPrecision = deltaToPrecision(prec.MinDelta);
    prec.MeanPrecision = deltaToPrecision(prec.MeanDelta);
    prec.MedianDelta = calcMedian(diff);
    prec.MedianPrecision = deltaToPrecision(prec.MedianDelta);

    // Assuming GetErrSTDSlotDomain and GetErrSTDCoeffDomain are implemented elsewhere
    // prec.STDFreq = GetErrSTDSlotDomain(valuesWant, valuesTest, params.DefaultScale());
    // prec.STDTime = GetErrSTDCoeffDomain(valuesWant, valuesTest, params.DefaultScale());

    return prec;
}



PrecisionStats GetPrecisionStats(const std::vector<double>& valuesTest, const std::vector<double>& valuesWant) {
    PrecisionStats prec;
    prec.MaxDelta = {0, 0, 0};
    prec.MinDelta = {1, 1, 1};
    prec.MeanDelta = {0, 0, 0};
    prec.cdfResol = 500;

    prec.RealDist.resize(prec.cdfResol);
    prec.ImagDist.resize(prec.cdfResol);
    prec.L2Dist.resize(prec.cdfResol);

    std::vector<double> precReal(valuesWant.size());
    std::vector<double> precImag(valuesWant.size());
    std::vector<double> precL2(valuesWant.size());

    std::vector<Stats> diff(valuesWant.size());

    for (size_t i = 0; i < valuesWant.size(); i++) {
        double deltaReal = std::abs(valuesTest[i] - valuesWant[i]);
        double deltaImag = 0;
        double deltaL2 = std::sqrt(deltaReal * deltaReal);
        precReal[i] = log2(1 / deltaReal);
        precImag[i] = log2(1 / deltaImag);
        precL2[i] = log2(1 / deltaL2);

        diff[i] = {deltaReal, deltaImag, deltaL2};

        prec.MeanDelta.Real += deltaReal;
        prec.MeanDelta.Imag += deltaImag;
        prec.MeanDelta.L2 += deltaL2;

        if (deltaReal > prec.MaxDelta.Real) {
            prec.MaxDelta.Real = deltaReal;
        }
        if (deltaImag > prec.MaxDelta.Imag) {
            prec.MaxDelta.Imag = deltaImag;
        }
        if (deltaL2 > prec.MaxDelta.L2) {
            prec.MaxDelta.L2 = deltaL2;
        }

        if (deltaReal < prec.MinDelta.Real) {
            prec.MinDelta.Real = deltaReal;
        }
        if (deltaImag < prec.MinDelta.Imag) {
            prec.MinDelta.Imag = deltaImag;
        }
        if (deltaL2 < prec.MinDelta.L2) {
            prec.MinDelta.L2 = deltaL2;
        }
    }

    prec.MeanDelta.Real /= valuesWant.size();
    prec.MeanDelta.Imag /= valuesWant.size();
    prec.MeanDelta.L2 /= valuesWant.size();

    calcCDF(precReal, prec.RealDist, prec.cdfResol);
    calcCDF(precImag, prec.ImagDist, prec.cdfResol);
    calcCDF(precL2, prec.L2Dist, prec.cdfResol);

    prec.MinPrecision = deltaToPrecision(prec.MaxDelta);
    prec.MaxPrecision = deltaToPrecision(prec.MinDelta);
    prec.MeanPrecision = deltaToPrecision(prec.MeanDelta);
    prec.MedianDelta = calcMedian(diff);
    prec.MedianPrecision = deltaToPrecision(prec.MedianDelta);

    // Assuming GetErrSTDSlotDomain and GetErrSTDCoeffDomain are implemented elsewhere
    // prec.STDFreq = GetErrSTDSlotDomain(valuesWant, valuesTest, params.DefaultScale());
    // prec.STDTime = GetErrSTDCoeffDomain(valuesWant, valuesTest, params.DefaultScale());

    return prec;
}