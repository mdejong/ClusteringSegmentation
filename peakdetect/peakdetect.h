#ifndef DETECT_PEAK_H
#define	DETECT_PEAK_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int detect_peak(
                const double*   data, // the data
                int             data_count, // row count of data
                int*            emi_peaks, // emission peaks will be put here
                int*            num_emi_peaks, // number of emission peaks found
                int             max_emi_peaks, // maximum number of emission peaks
                int*            absop_peaks, // absorption peaks will be put here
                int*            num_absop_peaks, // number of absorption peaks found
                int             max_absop_peaks, // maximum number of absorption peaks
                double          delta, // delta used for distinguishing peaks
                int             emi_first // should we search emission peak first of
                                          // absorption peak first?
  );

#ifdef __cplusplus
} // end extern C
#endif // __cplusplus

#endif // DETECT_PEAK_H
