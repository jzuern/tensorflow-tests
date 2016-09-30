#ifndef PERMUTOHEDRAL_LATTICE_H
#define PERMUTOHEDRAL_LATTICE_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <sys/time.h>


using namespace std;

/***************************************************************/
/* Hash table implementation for permutohedral lattice
 *
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (d+1)-
 * dimensional space.
 */
/***************************************************************/
class HashTablePermutohedral {
public:
        /* Constructor
         *  kd_: the dimensionality of the position vectors on the hyperplane.
         *  vd_: the dimensionality of the value vectors
         */
        HashTablePermutohedral(int kd_, int vd_) : kd(kd_), vd(vd_) {
                capacity = 1 << 15;
                filled = 0;
                entries = new Entry[capacity];
                keys = new short[kd*capacity/2];
                values = new float[vd*capacity/2];
                memset(values, 0, sizeof(float)*vd*capacity/2);
        }

        // Returns the number of vectors stored.
        int size() {
                return filled;
        }

        // Returns a pointer to the keys array.
        short *getKeys() {
                return keys;
        }

        // Returns a pointer to the values array.
        float *getValues() {
                return values;
        }

        /* Returns the index into the hash table for a given key.
         *     key: a pointer to the position vector.
         *       h: hash of the position vector.
         *  create: a flag specifying whether an entry should be created,
         *          should an entry with the given key not found.
         */
        int lookupOffset(short *key, size_t h, bool create = true) {

                // Double hash table size if necessary
                if (filled >= (capacity/2)-1) { grow(); }

                // Find the entry with the given key
                while (1) {
                        Entry e = entries[h];
                        // check if the cell is empty
                        if (e.keyIdx == -1) {
                                if (!create) return -1; // Return not found.
                                // need to create an entry. Store the given key.
                                for (int i = 0; i < kd; i++)
                                        keys[filled*kd+i] = key[i];
                                e.keyIdx = filled*kd;
                                e.valueIdx = filled*vd;
                                entries[h] = e;
                                filled++;
                                return e.valueIdx;
                        }

                        // check if the cell has a matching key
                        bool match = true;
                        for (int i = 0; i < kd && match; i++)
                                match = keys[e.keyIdx+i] == key[i];
                        if (match)
                                return e.valueIdx;

                        // increment the bucket with wraparound
                        h++;
                        if (h == capacity) h = 0;
                }
        }

        /* Looks up the value vector associated with a given key vector.
         *        k : pointer to the key vector to be looked up.
         *   create : true if a non-existing key should be created.
         */
        float *lookup(short *k, bool create = true) {
                size_t h = hash(k) % capacity;
                int offset = lookupOffset(k, h, create);
                if (offset < 0) return NULL;
                else return values + offset;
        };

        /* Hash function used in this implementation. A simple base conversion. */
        size_t hash(const short *key) {
                size_t k = 0;
                for (int i = 0; i < kd; i++) {
                        k += key[i];
                        k *= 2531011;
                }
                return k;
        }

private:
        /* Grows the size of the hash table */
        void grow() {
                printf("Resizing hash table\n");

                size_t oldCapacity = capacity;
                capacity *= 2;

                // Migrate the value vectors.
                float *newValues = new float[vd*capacity/2];
                memset(newValues, 0, sizeof(float)*vd*capacity/2);
                memcpy(newValues, values, sizeof(float)*vd*filled);
                delete[] values;
                values = newValues;

                // Migrate the key vectors.
                short *newKeys = new short[kd*capacity/2];
                memcpy(newKeys, keys, sizeof(short)*kd*filled);
                delete[] keys;
                keys = newKeys;

                Entry *newEntries = new Entry[capacity];

                // Migrate the table of indices.
                for (size_t i = 0; i < oldCapacity; i++) {
                        if (entries[i].keyIdx == -1) continue;
                        size_t h = hash(keys + entries[i].keyIdx) % capacity;
                        while (newEntries[h].keyIdx != -1) {
                                h++;
                                if (h == capacity) h = 0;
                        }
                        newEntries[h] = entries[i];
                }
                delete[] entries;
                entries = newEntries;
        }

        // Private struct for the hash table entries.
        struct Entry {
                Entry() : keyIdx(-1), valueIdx(-1) {
                }
                int keyIdx;
                int valueIdx;
        };

        short *keys;
        float *values;
        Entry *entries;
        size_t capacity, filled;
        int kd, vd;
};

/***************************************************************/
/* The algorithm class that performs the filter
 *
 * PermutohedralLattice::filter(...) does all the work.
 *
 */
/***************************************************************/
class PermutohedralLattice {
public:

        /* Filters given image against a reference image.
         *   im : image to be bilateral-filtered.
         *  ref : reference image whose edges are to be respected.
         */


        static void filter(tensorflow::Tensor image, tensorflow::Tensor ref, tensorflow::Tensor* out) {

          // cast tensors of type TF::Tensor to EIGEN::Tensor types
          auto out_eigen       = out->tensor<float,3>();
          auto image_eigen     = image.tensor<float,3>();
          auto ref_eigen       = ref.tensor<float,3>();

          int nChannels = 3;


          timeval t[5];

          // Create lattice
          gettimeofday(t+0, NULL);
          PermutohedralLattice lattice(nChannels, nChannels+1, image_eigen.dimension(0)*image_eigen.dimension(1)*1);

          // Splat into the lattice
          gettimeofday(t+1, NULL); printf("Splatting...\n");

          float *col = new float[nChannels+1];
          col[nChannels] = 1; // homogeneous coordinate



                  for (int y = 0; y < image_eigen.dimension(0); y++) {
                          for (int x = 0; x < image_eigen.dimension(1); x++) {
                                  for (int c = 0; c < nChannels; c++) {
                                          col[c] = image_eigen(x,y,c);
                                  }

                                  std::vector<float> a(3); // TODO: this is terrible
                                  for(int i = 0; i < 3; i++) a[i] = ref_eigen(x,y,i);
                                  // refPtr is pointing to ref image entries ref(x,y,<all>)
                                  lattice.splat(a, col);
                          }
                  }


          // Blur the lattice
          gettimeofday(t+2, NULL); printf("Blurring...");
          lattice.blur();

          // Slice from the lattice
          gettimeofday(t+3, NULL); printf("Slicing...\n");
          // Eigen::Tensor<float,3> out(image_eigen.dimension(0), image_eigen.dimension(1), nChannels);

          lattice.beginSlice();
          for (int y = 0; y < image_eigen.dimension(0); y++) {
                    for (int x = 0; x < image_eigen.dimension(1); x++) {
                            lattice.slice(col);
                            float scale = 1.0f/col[3];
                            for (int c = 0; c < nChannels; c++) {
                                    out_eigen(x,y,c) = col[c]*scale;
                            }
                    }
            }

          // Print time elapsed for each step
          gettimeofday(t+4, NULL);
          const char *names[4] = {"Init  ", "Splat ", "Blur  ", "Slice "};
          for (int i = 1; i < 5; i++)
                  printf("%s: %3.3f ms\n", names[i-1], (t[i].tv_sec - t[i-1].tv_sec) +
                          (t[i].tv_usec - t[i-1].tv_usec)/1000000.0);

        }

        /* Constructor
         *     d_ : dimensionality of key vectors
         *    vd_ : dimensionality of value vectors
         * nData_ : number of points in the input
         */
        PermutohedralLattice(int d_, int vd_, int nData_) :
                d(d_), vd(vd_), nData(nData_), hashTable(d_, vd_) {

                // Allocate storage for various arrays
                elevated = new float[d+1];
                scaleFactor = new float[d];

                greedy = new short[d+1];
                rank = new char[d+1];
                barycentric = new float[d+2];
                replay = new ReplayEntry[nData*(d+1)];
                nReplay = 0;
                canonical = new short[(d+1)*(d+1)];
                key = new short[d+1];

                // compute the coordinates of the canonical simplex, in which
                // the difference between a contained point and the zero
                // remainder vertex is always in ascending order. (See pg.4 of paper.)
                for (int i = 0; i <= d; i++) {
                        for (int j = 0; j <= d-i; j++)
                                canonical[i*(d+1)+j] = i;
                        for (int j = d-i+1; j <= d; j++)
                                canonical[i*(d+1)+j] = i - (d+1);
                }

                // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
                for (int i = 0; i < d; i++) {
                        // the diagonal entries for normalization
                        scaleFactor[i] = 1.0f/(sqrtf((float)(i+1)*(i+2)));

                        /* We presume that the user would like to do a Gaussian blur of standard deviation
                         * 1 in each dimension (or a total variance of d, summed over dimensions.)
                         * Because the total variance of the blur performed by this algorithm is not d,
                         * we must scale the space to offset this.
                         *
                         * The total variance of the algorithm is (See pg.6 and 10 of paper):
                         *  [variance of splatting] + [variance of blurring] + [variance of splatting]
                         *   = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
                         *   = 2d(d+1)(d+1)/3.
                         *
                         * So we need to scale the space by (d+1)sqrt(2/3).
                         */
                        scaleFactor[i] *= (d+1)*sqrtf(2.0/3);
                }
        }


        /* Performs splatting with given position and value vectors */
        void splat(std::vector<float> position, float *value) {

                // first rotate position into the (d+1)-dimensional hyperplane
                elevated[d] = -d*position[d-1]*scaleFactor[d-1];
                for (int i = d-1; i > 0; i--)
                        elevated[i] = (elevated[i+1] -
                                       i*position[i-1]*scaleFactor[i-1] +
                                       (i+2)*position[i]*scaleFactor[i]);
                elevated[0] = elevated[1] + 2*position[0]*scaleFactor[0];

                // prepare to find the closest lattice points
                float scale = 1.0f/(d+1);
                char * myrank = rank;
                short * mygreedy = greedy;

                // greedily search for the closest zero-colored lattice point
                int sum = 0;
                for (int i = 0; i <= d; i++) {
                        float v = elevated[i]*scale;
                        float up = ceilf(v)*(d+1);
                        float down = floorf(v)*(d+1);

                        if (up - elevated[i] < elevated[i] - down) mygreedy[i] = (short)up;
                        else mygreedy[i] = (short)down;

                        sum += mygreedy[i];
                }
                sum /= d+1;

                // rank differential to find the permutation between this simplex and the canonical one.
                // (See pg. 3-4 in paper.)
                memset(myrank, 0, sizeof(char)*(d+1));
                for (int i = 0; i < d; i++)
                        for (int j = i+1; j <= d; j++)
                                if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j]) myrank[i]++; else myrank[j]++;

                if (sum > 0) {
                        // sum too large - the point is off the hyperplane.
                        // need to bring down the ones with the smallest differential
                        for (int i = 0; i <= d; i++) {
                                if (myrank[i] >= d + 1 - sum) {
                                        mygreedy[i] -= d+1;
                                        myrank[i] += sum - (d+1);
                                } else
                                        myrank[i] += sum;
                        }
                } else if (sum < 0) {
                        // sum too small - the point is off the hyperplane
                        // need to bring up the ones with largest differential
                        for (int i = 0; i <= d; i++) {
                                if (myrank[i] < -sum) {
                                        mygreedy[i] += d+1;
                                        myrank[i] += (d+1) + sum;
                                } else
                                        myrank[i] += sum;
                        }
                }

                // Compute barycentric coordinates (See pg.10 of paper.)
                memset(barycentric, 0, sizeof(float)*(d+2));
                for (int i = 0; i <= d; i++) {
                        barycentric[d-myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
                        barycentric[d+1-myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
                }
                barycentric[0] += 1.0f + barycentric[d+1];

                // Splat the value into each vertex of the simplex, with barycentric weights.
                for (int remainder = 0; remainder <= d; remainder++) {
                        // Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
                        for (int i = 0; i < d; i++)
                                key[i] = mygreedy[i] + canonical[remainder*(d+1) + myrank[i]];

                        // Retrieve pointer to the value at this vertex.
                        float * val = hashTable.lookup(key, true);

                        // Accumulate values with barycentric weight.
                        for (int i = 0; i < vd; i++)
                                val[i] += barycentric[remainder]*value[i];

                        // Record this interaction to use later when slicing
                        replay[nReplay].offset = val - hashTable.getValues();
                        replay[nReplay].weight = barycentric[remainder];
                        nReplay++;

                }
        }

        // Prepare for slicing
        void beginSlice() {
                nReplay = 0;
        }

        /* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
         * containing each position vector were calculated and stored in the splatting step.
         * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
         */
        void slice(float *col) {
                float *base = hashTable.getValues();
                for (int j = 0; j < vd; j++) col[j] = 0;
                for (int i = 0; i <= d; i++) {
                        ReplayEntry r = replay[nReplay++];
                        for (int j = 0; j < vd; j++) {
                                col[j] += r.weight*base[r.offset + j];
                        }
                }
        }

        /* Performs a Gaussian blur along each projected axis in the hyperplane. */
        void blur() {
                // Prepare arrays
                short *neighbor1 = new short[d+1];
                short *neighbor2 = new short[d+1];
                float *newValue = new float[vd*hashTable.size()];
                float *oldValue = hashTable.getValues();
                float *hashTableBase = oldValue;

                float *zero = new float[vd];
                for (int k = 0; k < vd; k++) zero[k] = 0;

                // For each of d+1 axes,
                for (int j = 0; j <= d; j++) {
                        printf(" %d", j); fflush(stdout);

                        // For each vertex in the lattice,
                        for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j
                                short *key    = hashTable.getKeys() + i*(d);// keys to current vertex
                                for (int k = 0; k < d; k++) {
                                        neighbor1[k] = key[k] + 1;
                                        neighbor2[k] = key[k] - 1;
                                }
                                neighbor1[j] = key[j] - d;
                                neighbor2[j] = key[j] + d; // keys to the neighbors along the given axis.

                                float *oldVal = oldValue + i*vd;
                                float *newVal = newValue + i*vd;

                                float *vm1, *vp1;

                                vm1 = hashTable.lookup(neighbor1, false); // look up first neighbor
                                if (vm1) vm1 = vm1 - hashTableBase + oldValue;
                                else vm1 = zero;

                                vp1 = hashTable.lookup(neighbor2, false); // look up second neighbor
                                if (vp1) vp1 = vp1 - hashTableBase + oldValue;
                                else vp1 = zero;

                                // Mix values of the three vertices
                                for (int k = 0; k < vd; k++)
                                        newVal[k] = (0.25f*vm1[k] + 0.5f*oldVal[k] + 0.25f*vp1[k]);
                        }
                        float *tmp = newValue;
                        newValue = oldValue;
                        oldValue = tmp;
                        // the freshest data is now in oldValue, and newValue is ready to be written over
                }

                // depending where we ended up, we may have to copy data
                if (oldValue != hashTableBase) {
                        memcpy(hashTableBase, oldValue, hashTable.size()*vd*sizeof(float));
                        delete oldValue;
                } else {
                        delete newValue;
                }
                printf("\n");

                delete zero;
                delete neighbor1;
                delete neighbor2;
        }

private:

        int d, vd, nData;
        float *elevated, *scaleFactor, *barycentric;
        short *canonical;
        short *key;

        // slicing is done by replaying splatting (ie storing the sparse matrix)
        struct ReplayEntry {
                int offset;
                float weight;
        } *replay;
        int nReplay, nReplaySub;

public:
        char  *rank;
        short *greedy;
        HashTablePermutohedral hashTable;
};

#endif // PERMUTOHEDRAL_LATTICE_H