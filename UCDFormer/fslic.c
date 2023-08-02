#include "Python.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#define IDX(x, y, c, width, depth) (((y) * (width) * (depth)) + ((x) * (depth)) + (c))
#define IDX2(x, y, width) (((y) * (width)) + (x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define VALID(x, y, w, h) (((x)>=0 && (x)<w && (y)>=0 && (y)<h) ? 1 : 0)

void free_1d(int* ptr){
    free(ptr);
}

void free_1d_d(double* ptr){
    free(ptr);
}

void free_2d(int** ptr, int h){
    for(int i=0; i<h; ++i){
        free(ptr[i]);
    }
    free(ptr);
}

void free_2d_d(double** ptr, int h){
    for(int i=0; i<h; ++i){
        free(ptr[i]);
    }
    free(ptr);
}

void free_3d(int*** ptr, int h, int w){
    for(int i=0; i<h; ++i){
        for(int j=0; j<w; ++j){
            free(ptr[i][j]);
        }
    }

    for(int i=0; i<h; ++i){
        free(ptr[i]);
    }

    free(ptr);
}

void free_3d_d(double*** ptr, int h, int w){
    for(int i=0; i<h; ++i){
        for(int j=0; j<w; ++j){
            free(ptr[i][j]);
        }
    }

    for(int i=0; i<h; ++i){
        free(ptr[i]);
    }

    free(ptr);
}

int contains(int* arr, int val, int size){
    for(int i=0; i<size; ++i){
        if(arr[i] == val){
            return 1;
        }
    }
    return 0;
}

int contains_2d(int** labels, int h, int w, int k){
    for(int i=0; i<h; ++i){
        for(int j=0; j<w; ++j){
            if(labels[i][j] == k){
                return 1;
            }
        }
    }
    return 0;
}


double* list_to_array(PyObject* incoming) {
    double* data;
    data = malloc(PyList_Size(incoming) * sizeof(double));
    for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
        PyObject *value = PyList_GetItem(incoming, i);
        data[i] = PyFloat_AS_DOUBLE(value);
    }
    return data;
}

int*** create3darray(int h, int w, int depth, int val){
    int*** result = malloc(sizeof(int**) * h);
    for(int i=0; i<h; ++i){
        result[i] = malloc(sizeof(int*) * w);
        for(int j=0; j<w; ++j){
            result[i][j] = calloc(depth, sizeof(int));
            for(int z=0; z<depth; ++z){
                result[i][j][z] = val;
            }
        }
    }
    return result;
}

double*** create3darray_d(int h, int w, int depth, double val){
    double*** result = malloc(sizeof(double**) * h);
    for(int i=0; i<h; ++i){
        result[i] = malloc(sizeof(double*) * w);
        for(int j=0; j<w; ++j){
            result[i][j] = calloc(depth, sizeof(double));
            for(int z=0; z<depth; ++z){
                result[i][j][z] = val;
            }
        }
    }
    return result;
}

int** create2darray(int h, int w, int val){
    int** result = malloc(sizeof(int*) * h);
    for(int i=0; i<h; ++i){
        result[i] = malloc(w * sizeof(int));
        for(int j=0; j<w; ++j){
            result[i][j] = val;
        }
    }
    return result;
}

double** create2darray_d(int h, int w, double val){
    double** result = malloc(sizeof(double*) * h);
    for(int i=0; i<h; ++i){
        result[i] = malloc(w * sizeof(double));
        for(int j=0; j<w; ++j){
            result[i][j] = val;
        }
    }
    return result;
}

double*** array_to_3d(double* list, int w, int h, int depth){
    double*** img = create3darray_d(h, w, depth, 0.0);
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            for(int z=0; z<depth; ++z){
                img[y][x][z] = list[IDX(x, y, z, w, depth)];
            }
        }
    }
    free_1d_d(list);
    return img;
}

float gradient(int x, int y, double*** img, int w, int h){
    float grad = 0.0;

    int x1 = MAX(0, x-1);
    int x2 = MIN(w, x+1);
    int y1 = MAX(0, y-1);
    int y2 = MIN(h, y+1);

    float diff_x = (img[y][x1][0] - img[y][x2][0]) * (img[y][x1][0] - img[y][x2][0]) + 
                   (img[y][x1][1] - img[y][x2][1]) * (img[y][x1][1] - img[y][x2][1]) + 
                   (img[y][x1][2] - img[y][x2][2]) * (img[y][x1][2] - img[y][x2][2]);

    float diff_y = (img[y1][x][0] - img[y2][x][0]) * (img[y1][x][0] - img[y2][x][0]) + 
                   (img[y1][x][1] - img[y2][x][1]) * (img[y1][x][1] - img[y2][x][1]) + 
                   (img[y1][x][2] - img[y2][x][2]) * (img[y1][x][2] - img[y2][x][2]);

    grad = diff_x + diff_y;
    return grad;
}

void correct_centers(double*** img, int w, int h, double** centers, int num_clusters){
    for(int k=0; k<num_clusters; ++k){
        float min_gradient = DBL_MAX;
        int idx_y = -1;
        int idx_x = -1;
        int c_x = centers[k][0];
        int c_y = centers[k][1];
        for(int ny=c_y-1; ny<c_y+2; ++ny){
            for(int nx=c_x-1; nx<c_x+2; ++nx){
                float grad = gradient(nx, ny, img, w, h);
                if(grad < min_gradient){
                    min_gradient = grad;
                    idx_x = nx;
                    idx_y = ny;
                }
            }
        }
        centers[k][0] = idx_x;
        centers[k][1] = idx_y;
        centers[k][2] = img[idx_y][idx_x][0];
        centers[k][3] = img[idx_y][idx_x][1];
        centers[k][4] = img[idx_y][idx_x][2];
    }
}

double** initialize_centers(double*** img, int w, int h, int depth, int* num_clusters, int* grid_s, int m){
    // initial number of superpixels

    int grid_size = sqrt((double)(h * w) / (double)(m));
    if(grid_size  < 1){
        grid_size = 1;
    }

    *grid_s = grid_size;

    int x_ = w/grid_size;
    int y_ = h/grid_size;
    int extra_x = w - grid_size * x_;
    int extra_y = h - grid_size * y_;
    double per_grid_extra_x = 0.0, per_grid_extra_y = 0.0;
    if(extra_x != 0){
        per_grid_extra_x = (double) extra_x / (double) x_;
    }
    if(extra_y != 0){
        per_grid_extra_y = (double) extra_y / (double) y_;
    }
    *num_clusters = x_ * y_;
    double** centers = create2darray_d(*num_clusters, (2+depth), 0);

    int i = 0;
    for(int y = 0; y < (int)(h-(double)grid_size/2.0 - extra_y); y+=grid_size){

        int new_y = y + (grid_size / 2) + per_grid_extra_y * (double)(y / grid_size);

        for(int x = 0; x < (int)(w-(double)grid_size/2.0 - extra_x); x+=grid_size){

            int new_x = x + (grid_size / 2) + per_grid_extra_x * (x / grid_size);

            centers[i][0] = new_x;
            centers[i][1] = new_y;
            for(int z=0; z<depth; ++z){
                centers[i][z+2] = img[new_y][new_x][z];
            }
            i++;
        }
    }

//     correct_centers(img, w, h, centers, *num_clusters);

    return centers;
}

void enforce_connectivity(int** labels, double** centers, int w, int h, int num_clusters){
    int avg_superpixel_size = h*w/num_clusters;
    int** queue = create2darray(avg_superpixel_size * 10, 2, 0);
    int** mini_queue = create2darray(avg_superpixel_size * 5, 2, 0);
    int first, last, mini_first, mini_last;
    int** visited = create2darray(h, w, 0);
    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, -1, 1, 0};
    for(int k=0; k<num_clusters; ++k){
        // Check if cluster is represented at all in labels
        if(!contains_2d(labels, h, w, k)){
            continue;
        }

        // Get center of main cluster region
        int c_x = centers[k][0];
        int c_y = centers[k][1];

        //~ if(labels[c_y][c_x] != k){
            //~ continue;
        //~ }

        // Perform region growing on that cluster
        first = 0;
        last = 1;

        queue[first][0] = c_x;
        queue[first][1] = c_y;

        while(first != last){
            int popped_x = queue[first][0];
            int popped_y = queue[first][1];
            first++;

            for(int delta=0; delta<4; delta++){
                int current_x = popped_x + dx[delta];
                int current_y = popped_y + dy[delta];
                if(VALID(current_x, current_y, w, h)){
                    if(k==labels[current_y][current_x] && !visited[current_y][current_x]){
                        visited[current_y][current_x] = 1;
                        queue[last][0] = current_x;
                        queue[last][1] = current_y;
                        last++;
                    }
                }
            }
        }

        // Get pixel of same cluster but not in the main region (not visited)
        last = 0;
        for(int i=0; i<h; ++i){
            for(int j=0; j<w; ++j){
                if(labels[i][j]==k && !visited[i][j]){
                    queue[last][0] = j;
                    queue[last][1] = i;
                    last++;
                }
            }
        }
        // For each pixel, perform region growing and color it by its first cluster (make sure to mark the current small region as visited
        // since there could be multiple small regions)
        int last_neighbour = -1;
        for(int i=0; i<last; ++i){
            int popped_x = queue[i][0];
            int popped_y = queue[i][1];
            if(visited[popped_y][popped_x]){
                continue;
            }

            mini_queue[0][0] = popped_x;
            mini_queue[0][1] = popped_y;
            mini_first = 0;
            mini_last = 1;

            while(mini_first!=mini_last){
                popped_x = mini_queue[mini_first][0];
                popped_y = mini_queue[mini_first][1];
                mini_first++;

                for(int delta=0; delta<4; delta++){
                    int current_x = popped_x + dx[delta];
                    int current_y = popped_y + dy[delta];
                    if(VALID(current_x, current_y, w, h)){
                        if(k==labels[current_y][current_x] && !visited[current_y][current_x]){
                            visited[current_y][current_x] = 1;
                            mini_queue[mini_last][0] = current_x;
                            mini_queue[mini_last][1] = current_y;
                            mini_last++;
                        }else if(k!=labels[current_y][current_x]){
                            last_neighbour = labels[current_y][current_x];
                        }
                    }
                }
            }

            for(int j=0; j<mini_last; ++j){
                labels[mini_queue[j][1]][mini_queue[j][0]] = last_neighbour;
            }
        }
        // Reset visited array
        for(int i=0; i<h; ++i){
            for(int j=0; j<w; ++j){
                visited[i][j] = 0;
            }
        }
    }
}

// takes in 5 dimensional vectors (x, y, c1, c2, c3) and returns squared distance in color space.
double distance_slic(double v1[], double v2[], int depth, int grid_size, double compactness){
    double diff0 = v1[0]-v2[0];
    double diff1 = v1[1]-v2[1];
    double distance_space = sqrt((diff0 * diff0) + (diff1 * diff1));
    double distance_feature = 0.0;
    for(int z=0; z<depth; ++z){
        distance_feature += (v1[z+2] - v2[z+2]) * (v1[z+2] - v2[z+2]);
    }
    distance_feature = sqrt(distance_feature);
    double result = distance_feature + (distance_space * compactness / grid_size);
    return result;
}

double* fslic(PyObject* arg, int w, int h, int depth, int m, double compactness, int max_iterations, double p, double q){
    double* list = list_to_array(arg);
    double*** img = array_to_3d(list, w , h, depth);
    int num_clusters;
    int grid_size;
    double** centers = initialize_centers(img, w, h, depth, &num_clusters, &grid_size, m);

    // Initialize labels
    int num_of_possible_clusters = 8;
    int*** G = create3darray(h, w, num_of_possible_clusters, -1);
    double*** D = create3darray_d(h, w, num_of_possible_clusters, DBL_MAX);

    double fuzziness = 2.0;
    double exponent = 1.0 / (fuzziness - 1.0); 

    // Membership matrix 
    double*** U = create3darray_d(h, w, num_of_possible_clusters, 0);

    // Neighbourhood's membership matrix
    double*** H = create3darray_d(h, w, num_of_possible_clusters, 0);

    // Advanced Membership matrix 
    double*** U_ = create3darray_d(h, w, num_of_possible_clusters, 0);

    // Visit control
    int** F = create2darray(h, w, 0);

    for(int iterations = 0; iterations < max_iterations; ++iterations){

        // MEMSET NOT CREATE
        for(int y=0; y<h; ++y){
            memset(F[y], 0, w * sizeof(int));
            for(int x=0; x<w; ++x){
                memset(U[y][x], 0, num_of_possible_clusters * sizeof(double));
                memset(H[y][x], 0, num_of_possible_clusters * sizeof(double));
                memset(U_[y][x], 0, num_of_possible_clusters * sizeof(double));
            }
        }

        // Loop through all clusters
        for(int k = 0; k < num_clusters; ++k){

            // Get search region of super-pixel
            int x_c = centers[k][0];
            int y_c = centers[k][1];

            int x1 = MAX(0, x_c - grid_size);
            int x2 = MIN(w, x_c + grid_size + 1);
            int y1 = MAX(0, y_c - grid_size);
            int y2 = MIN(h, y_c + grid_size + 1);

            // Loop through pixels in the search region
            for(int y=y1; y<y2; ++y){
                for(int x=x1; x<x2; ++x){

                    // Get distance to cluster center
                    double c[depth+2];
                    c[0] = x;
                    c[1] = y;

                    for(int z=0; z<depth; ++z){
                        c[z+2] = img[y][x][z];
                    }

                    double distance = distance_slic(centers[k], c, depth, grid_size, compactness);

                    // Check if the pixel has already been visited 3 times in this iterations
                    if(F[y][x] < num_of_possible_clusters){
                        G[y][x][F[y][x]] = k;
                        D[y][x][F[y][x]] = distance;
                        F[y][x]++;
                    }
                    else {
                        double max_d = D[y][x][0];
                        int ix = 0;
                        for(int z=1; z<num_of_possible_clusters; ++z){
                            if(D[y][x][z] > max_d){
                                max_d = D[y][x][z];
                                ix = z;
                            }
                        }
                        if(distance < max_d){
                            // Replace maximum distance with new distance and update pixel labels
                            G[y][x][ix] = k;
                            D[y][x][ix] = distance;
                        }

                    }
                }
            }
        }

        // Update membership matrix U
        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){

                int flag = 0;
                int l;
                for(l=0; l<F[y][x]; ++l){
                    if(D[y][x][l] == 0.0){
                        flag = 1;
                        break;
                    }
                }
                if(flag == 1){
                    U[y][x][l] = 1.0;
                    continue;
                }

                double total = 0.0;
                for(int m=0; m<F[y][x]; ++m){
                    total += 1.0 / (pow(D[y][x][m], exponent));
                }
                for(int m=0; m<F[y][x]; ++m){
                    double d_ = pow(D[y][x][m], exponent);
                    U[y][x][m] = 1.0 / (d_ * total);
                }
            }
        }

        // Include neighbourhood's membership (matrix H)
        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){

                int x1 = MAX(0, x - 1);
                int x2 = MIN(w, x + 2);
                int y1 = MAX(0, y - 1);
                int y2 = MIN(h, y + 2);

                for(int l=0; l<F[y][x]; ++l){
                    double total = 0.0;

                    // Loop though 8 neighoburs
                    for(int ny=y1; ny<y2; ++ny){
                        for(int nx=x1; nx<x2; ++nx){

                            // Skip current pixel
                            if(nx==x && ny==y){
                                continue;
                            }

                            for(int nl=0; nl<F[ny][nx]; ++nl){

                                //If labels match, add neigbour's membership to total
                                if(G[y][x][l] == G[ny][nx][nl]){
                                    total += U[ny][nx][nl];
                                    break;
                                }
                            }

                        }
                    }
                    H[y][x][l] = total;
                }
            }
        }

        // Update advanced membership matrix U_
        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){

                double total = 0.0;
                // Loop through pixels labels
                for(int l=0; l<F[y][x]; ++l){
                    total += pow(U[y][x][l], p) * pow(H[y][x][l], q);
                }

                for(int l=0; l<F[y][x]; ++l){
                    if(total == 0.0){
                        U_[y][x][l] = 0.0;
                    }else{
                        U_[y][x][l] = (pow(U[y][x][l], p) * pow(H[y][x][l], q)) / total;
                    }
                }
            }
        }

        // Update cluster centers
        for(int k = 0; k < num_clusters; ++k){

            double new_center[depth+2];

            for(int z=0; z<depth+2; ++z){
                new_center[z] = 0.0;
            }
            double denominator = 0.0;

            // Loop through all pixels
            for(int y=0; y<h; ++y){
                for(int x=0; x<w; ++x){

                    // Loop through all labels of each pixel
                    for(int l=0; l<F[y][x]; ++l){

                        // If pixel is a member of cluster, use its color to update the cluster center's color
                        if(G[y][x][l] == k){
                            double membership = pow(U_[y][x][l], fuzziness);
                            new_center[0] += membership * x;
                            new_center[1] += membership * y;
                            for(int z=0; z<depth; ++z){
                                new_center[z+2] += membership * img[y][x][z];
                            }
                            denominator += membership;
                        }
                    }
                }
            }

            for(int z=0; z<depth+2; ++z){
                centers[k][z] = new_center[z] /= denominator;
            }
        }

    }

    int** labels = create2darray(h, w, -1);
    int* cluster_sizes = calloc(num_clusters, sizeof(int));
    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            int idx = 0;
            double max_mem = U_[y][x][0];
            for(int z=1; z<num_of_possible_clusters; ++z){
                if(U_[y][x][z] > max_mem){
                    max_mem = U_[y][x][z];
                    idx = z;
                }
            }
            int k = G[y][x][idx];
            labels[y][x] = k;
            cluster_sizes[k]++;
        }
    }

    enforce_connectivity(labels, centers, w, h, num_clusters);

    double* L_1d = malloc(h * w * depth * sizeof(double));

    for(int y=0; y<h; ++y){
        for(int x=0; x<w; ++x){
            int k = labels[y][x];
            for(int z=0; z<depth; ++z){
                L_1d[IDX(x, y, z, w, depth)] = centers[k][z+2];
            }
        }
    }

    free_2d_d(centers, num_clusters);
    free_3d(G, h, w);
    free_3d_d(D, h, w);
    free_1d(cluster_sizes);
    free_2d(labels, h);
    free_3d_d(U, h, w);
    free_3d_d(H, h, w);
    free_3d_d(U_, h, w);
    free_2d(F, h);
    free_3d_d(img, h, w);

    return L_1d;
}