struct unionfind {
  unsigned int size;
  unsigned int count;
  unsigned int* weights;
  unsigned int* parents;
};

struct unionfind* unionfind_new(unsigned int size);
void unionfind_init(struct unionfind *uf);
unsigned int unionfind_find(struct unionfind *uf, unsigned int id);
unsigned int unionfind_union(struct unionfind *uf, unsigned int i1, unsigned int i2);
unsigned int unionfind_count(struct unionfind *uf);
void unionfind_delete(struct unionfind *uf);

