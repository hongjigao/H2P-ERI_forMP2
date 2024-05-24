#include <stdio.h>
#include <stdlib.h>

void insertionSort(int *key, double *val, size_t l, size_t r) {
    size_t i, j;
    int keyToInsert;
    double valToInsert;
    for (i = l + 1; i < r; i++) {
        keyToInsert = key[i];
        valToInsert = val[i];
        j = i;
        // Move elements of key[l..i-1] and val[l..i-1], that are
        // greater than keyToInsert, to one position ahead
        // of their current position
        while (j > l && key[j - 1] > keyToInsert) {
            key[j] = key[j - 1];
            val[j] = val[j - 1];
            j--;
        }
        key[j] = keyToInsert;
        val[j] = valToInsert;
    }
}

// Example usage
int main() {
    int key[] = {12, 11, 13, 5, 6, 7,6,6,1};
    double val[] = {1.2, 1.1, 1.3, 0.5, 0.6, 0.7,0.5,0.5,0.5};
    size_t n = sizeof(key)/sizeof(key[0]);

    // Sort the entire array
    insertionSort(key, val, 0, n);

    printf("Sorted array:\n");
    for (size_t i = 0; i < n; i++)
        printf("%d: %f\n", key[i], val[i]);
    return 0;
}