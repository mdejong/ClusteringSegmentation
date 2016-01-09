#ifdef __cplusplus
extern "C" {
#endif

struct my_pair {
  unsigned int r1;
  unsigned int r2;
  unsigned int diff;
};

struct srm {
  unsigned int width;
  unsigned int height;
  unsigned int size;
  unsigned int channels;
  uint8_t *in;
  uint8_t *out;
  unsigned int *sizes;
  double logdelta;
  unsigned int smallregion;
  double g;
  double Q;
  struct unionfind *uf;
  unsigned int borders;
  struct my_pair *pairs;
  unsigned int n_pairs;
  struct my_pair *ordered_pairs;
  unsigned int widthStep_in;
  unsigned int widthStep_out;
};

struct srm* srm_new(double Q, unsigned int width, unsigned int height, unsigned int channels, unsigned int borders);
void srm_run(struct srm *srm, unsigned int widthStep_in, uint8_t *in, unsigned int widthStep_out, uint8_t *out);
unsigned int srm_regions_count(struct srm *srm);
unsigned int* srm_regions(struct srm *srm);
unsigned int* srm_regions_sizes(struct srm *srm);
void srm_delete(struct srm *srm);

void SRM(double Q, unsigned int width, unsigned int height, unsigned int channels, uint8_t *in, uint8_t *out, unsigned int borders);

#ifdef __cplusplus
}
#endif