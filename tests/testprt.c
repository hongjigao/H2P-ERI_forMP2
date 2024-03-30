#include <stdio.h>
#include <stdlib.h>

void swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int *key, double *val, int low, int high) {
    int pivot = key[high];
    int i = low - 1;
    for (int j = low; j <= high - 1; j++) {
        if (key[j] < pivot) {
            i++;
            swap_int(&key[i], &key[j]);
            swap_double(&val[i], &val[j]);
        }
    }
    swap_int(&key[i + 1], &key[high]);
    swap_double(&val[i + 1], &val[high]);
    return (i + 1);
}

void Qsort_double_long(int *key, double *val, int low, int high) {
    if (low < high) {
        int pi = partition(key, val, low, high);

        Qsort_double_long(key, val, low, pi - 1);
        Qsort_double_long(key, val, pi + 1, high);
    }
}

// Example usage
int main() {
    int key[] = {10, 7, 8, 9, 1, 5,5,5,1};
    double val[] = {2.1, 1.2, 3.8, 4.9, 0.1, 5.5,5,2,1};
    int n = sizeof(key) / sizeof(key[0]);

    Qsort_double_long(key, val, 0, n-1);

    printf("Sorted arrays:\n");
    for (int i = 0; i < n; i++) {
        printf("%d: %f\n", key[i], val[i]);
    }

    return 0;
}