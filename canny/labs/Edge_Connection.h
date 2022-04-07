void threshold_detection(unsigned char *image, unsigned char *weak_img, unsigned char *edges_img, 
                        double thresh_high, int width, int height);

void edge_connection(unsigned char *weak_img, unsigned char *edge_img, int width, int height);
