//This module is used to read point clouds

/* The authors of this work have released all rights to it and placed it
in the public domain under the Creative Commons CC0 1.0 waiver
(http://creativecommons.org/publicdomain/zero/1.0/).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef pcl_hpp
#define pcl_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>

//Number of divisions for quad tree
#define SUBDIVISIONS 100

//A quad tree is generated that consists of segments defined by
struct index_t {
    long start, ende;
};

struct cloud_t {
    //coordinates are translated such that (offset_x, offset_y, 0) is now the origin
    std::vector<cv::Point3d> pointcloud;
    //Index for fast access
    struct index_t pktindex[SUBDIVISIONS+1][SUBDIVISIONS+1];
    //maximum and minimum coordinates (offsets subtracted) (used for access via index)
    double pxmin; double pxmax; double pymin; double pymax;
    double offset_x=0, offset_y=0;
};

void readLAS(const char * imagepath, cloud_t& pointcloud);
void readXYZ(const char * imagepath, cloud_t& pointcloud);

#endif /* pcl_hpp */
