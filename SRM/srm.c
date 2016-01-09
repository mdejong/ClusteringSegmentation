#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "unionfind.h"
#include "srm.h"

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

#define index(i, j) (i) * srm->width + (j)
#define offset(offset, idx, widthStep) {unsigned int _i = (idx) / srm->width; unsigned int _j = (idx) % srm->width; (offset) = _i * (widthStep) + srm->channels * _j;}

#define get_b(im, offset) (im)[(offset)    ]
#define get_g(im, offset) (im)[(offset) + 1]
#define get_r(im, offset) (im)[(offset) + 2]

#define set_b(im, offset, b) (im)[(offset)    ] = (b)
#define set_g(im, offset, g) (im)[(offset) + 1] = (g)
#define set_r(im, offset, r) (im)[(offset) + 2] = (r)

void initialize(struct srm *srm);
void segmentation(struct srm *srm);
void finalize(struct srm *srm);
void merge_small_regions(struct srm *srm);

unsigned int diff(struct srm *srm, unsigned int idx1, unsigned int idx2);
unsigned int merge_predicate(struct srm *srm, unsigned int reg1, unsigned int reg2);
void bucket_sort(struct my_pair *pairs, struct my_pair *ordered_pairs, unsigned int size);
void merge_regions(struct srm *srm, unsigned int r1, unsigned int r2);

void SRM(double Q, unsigned int width, unsigned int height, unsigned int channels, uint8_t *in, uint8_t *out, unsigned int borders) {
  struct srm *srm = srm_new(Q, width, height, channels, borders);
  srm_run(srm, width * channels, in, width * channels, out);
  srm_delete(srm);
}

struct srm* srm_new(double Q, unsigned int width, unsigned int height, unsigned int channels, unsigned int borders) {
  struct srm *srm;
  srm = malloc(sizeof(struct srm));

  srm->width         = width;
  srm->height        = height;
  srm->channels      = channels;
  srm->Q             = Q;
  srm->borders       = borders;

  srm->size          = width * height;
  srm->smallregion   = 0.001 * srm->size;

  srm->logdelta      = 2.0 * log(6.0 * srm->size);
  srm->g             = 256.0;
  srm->n_pairs       = 2 * (srm->width - 1) * (srm->height - 1) + (srm->height - 1) + (srm->width - 1);

  srm->uf            = unionfind_new(srm->size);
  srm->sizes         = malloc(srm->size * sizeof(unsigned int));
  srm->pairs         = malloc(srm->n_pairs * sizeof(struct my_pair));
  srm->ordered_pairs = malloc(srm->n_pairs * sizeof(struct my_pair));

  return srm;
}

void srm_run(struct srm *srm, unsigned int widthStep_in, uint8_t *in, unsigned int widthStep_out, uint8_t *out) {
  srm->in  = in;
  srm->widthStep_in = widthStep_in;
  srm->out = out;
  srm->widthStep_out = widthStep_out;

  initialize(srm);
  segmentation(srm);
  merge_small_regions(srm);
  finalize(srm);
}

unsigned int srm_regions_count(struct srm *srm) {
  return srm->uf->count;
}

unsigned int* srm_regions(struct srm *srm) {
  return srm->uf->parents;
}

unsigned int* srm_regions_sizes(struct srm *srm) {
  return srm->uf->weights;
}

void srm_delete(struct srm *srm) {
  unionfind_delete(srm->uf);
  free(srm->sizes);
  free(srm->pairs);
  free(srm->ordered_pairs);
  free(srm);
}

void initialize(struct srm *srm) {
  unionfind_init(srm->uf);

  memcpy(srm->out, srm->in, srm->channels * srm->size * sizeof(uint8_t));
  for (unsigned int i = 0; i < srm->size; i++) {
      unsigned int offset_in;
      offset(offset_in, i, srm->widthStep_in);

      unsigned int offset_out;
      offset(offset_out, i, srm->widthStep_out);
      set_r(srm->out, offset_out, get_r(srm->in, offset_in));
      set_g(srm->out, offset_out, get_g(srm->in, offset_in));
      set_b(srm->out, offset_out, get_b(srm->in, offset_in));

    srm->sizes[i] = 1;
  }
}

inline unsigned int diff(struct srm *srm, unsigned int idx1, unsigned int idx2) {
  unsigned int offset1;
  offset(offset1, idx1, srm->widthStep_in);
  unsigned char r1 = get_r(srm->in, offset1);
  unsigned char g1 = get_g(srm->in, offset1);
  unsigned char b1 = get_b(srm->in, offset1);

  unsigned int offset2;
  offset(offset2, idx2, srm->widthStep_in);
  unsigned char r2 = get_r(srm->in, offset2);
  unsigned char g2 = get_g(srm->in, offset2);
  unsigned char b2 = get_b(srm->in, offset2);

  unsigned int diff_r = r2 > r1 ? r2 - r1 : r1 - r2;
  unsigned int diff_g = g2 > g1 ? g2 - g1 : g1 - g2;
  unsigned int diff_b = b2 > b1 ? b2 - b1 : b1 - b2;

  return max(diff_r, max(diff_g, diff_b));
}

void segmentation(struct srm *srm) {
  // Consider C4-connectivity here

  unsigned int index;
  unsigned int pair_index = 0;
  for (unsigned int i = 0; i < srm->height - 1; i++) {
    for (unsigned int j = 0; j < srm->width - 1; j++) {
      index = index(i, j);

      // C4 left
      srm->pairs[pair_index].r1 = index;
      srm->pairs[pair_index].r2 = index + 1;
      srm->pairs[pair_index].diff = diff(srm, index, index + 1);
      pair_index++;

      // C4 below
      srm->pairs[pair_index].r1 = index;
      srm->pairs[pair_index].r2 = index + srm->width;
      srm->pairs[pair_index].diff = diff(srm, index, index + srm->width);
      pair_index++;
    }
  }

  // The two border lines
  for (unsigned int i = 0; i < srm->height - 1; i++) {
    index = index(i, srm->width - 1);

    srm->pairs[pair_index].r1 = index;
    srm->pairs[pair_index].r2 = index + srm->width;
    srm->pairs[pair_index].diff = diff(srm, index, index + srm->width);
    pair_index++;
  }
  for (unsigned int j = 0; j < srm->width - 1; j++) {
    index = index(srm->height - 1,  j);

    srm->pairs[pair_index].r1 = index;
    srm->pairs[pair_index].r2 = index + 1;
    srm->pairs[pair_index].diff = diff(srm, index, index + 1);
    pair_index++;
  }

  // Sorting the edges according to the maximum color channel difference
  bucket_sort(srm->pairs, srm->ordered_pairs, srm->n_pairs);

  // Merging similar regions
  unsigned int reg1, reg2;
  for (unsigned int i = 0; i < srm->n_pairs; i++) {
    reg1 = srm->ordered_pairs[i].r1;
    reg1 = unionfind_find(srm->uf, reg1);

    reg2 = srm->ordered_pairs[i].r2;
    reg2 = unionfind_find(srm->uf, reg2);

    if ((reg1 != reg2) && (merge_predicate(srm, reg1, reg2)))
      merge_regions(srm, reg1, reg2);
  }
}

unsigned int merge_predicate(struct srm *srm, unsigned int reg1, unsigned int reg2) {
  double dR, dG, dB;
  double logreg1, logreg2;
  double dev1, dev2, dev;

  unsigned int offset1;
  offset(offset1, reg1, srm->widthStep_out);
  unsigned int offset2;
  offset(offset2, reg2, srm->widthStep_out);

  dR = (double)get_r(srm->out, offset1) - (double)get_r(srm->out, offset2);
  dR *= dR;

  dG = (double)get_g(srm->out, offset1) - (double)get_g(srm->out, offset2);
  dG *= dG;

  dB = (double)get_b(srm->out, offset1) - (double)get_b(srm->out, offset2);
  dB *= dB;

  assert(reg1 < srm->size);
  /* printf("%i %i %i\n", reg1, srm->size, srm->sizes[reg1]); */
  assert(srm->sizes[reg1] != 0);
  logreg1 = min(srm->g, srm->sizes[reg1]) * log(1.0 + srm->sizes[reg1]);
  logreg2 = min(srm->g, srm->sizes[reg2]) * log(1.0 + srm->sizes[reg2]);

  dev1 = (srm->g * srm->g) / (2.0 * srm->Q * srm->sizes[reg1]) * (logreg1 + srm->logdelta);
  dev2 = (srm->g * srm->g) / (2.0 * srm->Q * srm->sizes[reg2]) * (logreg2 + srm->logdelta);

  dev = dev1 + dev2;

  return ((dR < dev) && (dG < dev) && (dB < dev));
}

void bucket_sort(struct my_pair *pairs, struct my_pair *ordered_pairs, unsigned int size) {
  unsigned int nbe[256];
  unsigned int cnbe[256];

  for (unsigned int i = 0; i < 256; i++)
    nbe[i] = 0;

  // class all elements according to their family
  for (unsigned int i = 0; i < size; i++)
    nbe[pairs[i].diff]++;

  // cumulative histogram
  cnbe[0] = 0;
  for (unsigned int i = 1; i < 256; i++)
    cnbe[i] = cnbe[i - 1] + nbe[i - 1]; // index of first element of category i

  // allocation
  for (unsigned int i = 0; i < size; i++) {
    ordered_pairs[cnbe[pairs[i].diff]++] = pairs[i];
  }
}

// Merge two regions
void merge_regions(struct srm *srm, unsigned int reg1, unsigned int reg2) {
  unsigned int reg = unionfind_union(srm->uf, reg1, reg2);

  unsigned int offset1;
  offset(offset1, reg1, srm->widthStep_out);
  unsigned int offset2;
  offset(offset2, reg2, srm->widthStep_out);
  unsigned int offset;
  offset(offset, reg, srm->widthStep_out);

  assert(reg1 < srm->size);
  assert(reg2 < srm->size);
  unsigned int new_size = srm->sizes[reg1] + srm->sizes[reg2];
  /* printf("tutu %i %i %i\n", new_size, srm->sizes[reg1], srm->sizes[reg2]); */
  double r_avg = (srm->sizes[reg1] * get_r(srm->out, offset1) + srm->sizes[reg2] * get_r(srm->out, offset2)) / new_size;
  double g_avg = (srm->sizes[reg1] * get_g(srm->out, offset1) + srm->sizes[reg2] * get_g(srm->out, offset2)) / new_size;
  double b_avg = (srm->sizes[reg1] * get_b(srm->out, offset1) + srm->sizes[reg2] * get_b(srm->out, offset2)) / new_size;
  /* printf("titi\n"); */

  srm->sizes[reg] = new_size;
  assert(srm->sizes[reg] != 0);
  set_r(srm->out, offset, (unsigned char)r_avg);
  set_g(srm->out, offset, (unsigned char)g_avg);
  set_b(srm->out, offset, (unsigned char)b_avg);
}

void merge_small_regions(struct srm *srm) {
  unsigned int reg1, reg2;

  unsigned int index;
  for (unsigned int i = 0; i < srm->height; i++) {
    for (unsigned int j = 1; j < srm->width; j++) {
      index = index(i, j);

      reg1 = unionfind_find(srm->uf, index);
      reg2 = unionfind_find(srm->uf, index - 1);

      if (reg1 != reg2) {
        if ((srm->sizes[reg1] < srm->smallregion) || (srm->sizes[reg2] < srm->smallregion))
          merge_regions(srm, reg1, reg2);
      }
    }
  }
}

#include <stdio.h>

void finalize(struct srm *srm) {
  unsigned int index, root;

  for (unsigned int i = 0; i < srm->height; i++) {
    for (unsigned int j = 0; j < srm->width; j++) {
      index = index(i, j);
      root = unionfind_find(srm->uf, index);
      unsigned int root_offset;
      offset(root_offset, root, srm->widthStep_out);

      unsigned int offset;
      offset(offset, index, srm->widthStep_out);
      set_r(srm->out, offset, get_r(srm->out, root_offset));
      set_g(srm->out, offset, get_g(srm->out, root_offset));
      set_b(srm->out, offset, get_b(srm->out, root_offset));
      
      if ((0)) {
        fprintf(stdout, "index %5d = 0x%02X%02X%02X\n", index, get_r(srm->out, root_offset), get_g(srm->out, root_offset) ,get_b(srm->out, root_offset));
      }
    }
  }
}

