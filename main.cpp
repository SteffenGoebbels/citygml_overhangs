//  Overhangs
//  =========
//  The tool adds roof overhangs to existing LoD2 city models in CityGML representation
//
//  Created by Steffen Goebbels on 11.05.2022, iPattern Institute, Faculty of Electrical Engineering and Computer Science, Niederrhein University of Applied Sciences, Krefeld, Germany
//  steffen.goebbels@hsnr.de

//  The algorithm is described in the paper
//  Steffen Goebbels, Regina Pohle-Froehlich: Automatic Reconstruction of Roof Overhangs for 3D City Models
//
//  Prerequisites: OpenCV and libCityGML (tested with version 2.3.0) libraries
//

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

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <citygml/citygml.h>
#include <citygml/citymodel.h>
#include <citygml/texture.h>
#include <citygml/address.h>
#include <citygml/polygon.h>

#include "median_cut.hpp"
#include "pcl.hpp"

//--------------------------------------------------------
//Global variables
//--------------------------------------------------------

std::string path_input; //Path of citygml file
std::string input_filename, output_filename = "";
std::string pcl_filename=""; //Filename of point cloud

//Parsed CityGML structure
std::shared_ptr<const citygml::CityModel> city;

//Point Cloud
struct cloud_t pointcloud;

struct edge_t {
    unsigned long start_vertex_3D; //Index of starting point
    unsigned long end_vertex_3D; //Index of end point
    unsigned long start_vertex_2D; //Index of starting point projected to ground plane
    unsigned long end_vertex_2D; //Index of end point projected to ground plane
    unsigned long polygon; //Index of corresponding polygon
    bool relevant; //does this edges requires a roof overhang?
    bool inside; //inside the footprint?
    double size; //size of overhang determined from gradient image of texture
    double size_from_cloud; //size of overhang determined from point cloud
    double size_from_color; //size of overhang determined from color comparison
    bool from_texture = false; //origin of overhang distance
    bool from_cloud = false;
    bool from_color = false;
    unsigned int LoD;
};

struct polygon_t {
    cv::Point3d normal;
    std::vector< unsigned long > edges;
    bool color_set = false;
    float r=0.4, g=0.4, b=0.4; //Color for all roof overhands of this roof facet
};

struct vertex_2D_t {
    cv::Point2d point; //vertices are handled in 2D to assess the roof topology
    std::vector< unsigned long > arriving_edges;
    std::vector< unsigned long > departing_edges;
    
};

struct footprint_edge_t {
    cv::Point2d left, right;
    double left_z, right_z; //Maximum z value for left and right point
    cv::Point2d direction;
};

#define WALL_VERTEX 0
#define INTERSECTION_VERTEX 1
#define PROJECTION_VERTEX 2

struct overhang_vertex_t {
    int type; //see above
    unsigned int LoD;
    cv::Point3d point;
};

struct overhang_2Dvertex_t {
    int type; //see above
    cv::Point2d point;
};
                         
#define BUILDING 0
#define BUILDING_PART 1

struct overhangs_of_roof_facet_t {
    std::vector< std::vector<overhang_vertex_t> > points;
    bool color_set = false;
    float r=0.4, g=0.4, b=0.4; //Color for all roof overhands of this roof facet
    
};

struct buildingOverhangs_t {
    std::string building_id;
    std::string buildingPart_id;
    //For each roof polygon: vector of overhang polygons
    std::vector < struct overhangs_of_roof_facet_t > roof_facet;
    int type; //Building or BUILDING_PART
};

struct farbe_t {
    float r,g,b; //between 0 and 1
    std::vector<std::string> id;
};

//--------------------------------------
// functions
//--------------------------------------

//Read CityGML model
std::shared_ptr<const citygml::CityModel> readModel() {

    std::shared_ptr<const citygml::CityModel> city;
    citygml::ParserParams params;
    
    params.pruneEmptyObjects=true;
    params.keepVertices=true;
    params.optimize=true;
    
    try{
        city = citygml::load( input_filename.c_str(), params );
    }catch(...){
        std::cout << "Parser error, check if input filename is correct." << std::endl;
        return NULL;
    }
    return city;
}

//compute outer normal of a polygon
cv::Point3d computeNormal(std::vector<cv::Point3d> &polygon) {
    
    int k,l;
    cv::Point3d normal(0,0,0);
    
    //Select three vertices:
    //Search to vertices with a largest mutual distance. Then add a third vertex such that a determinant becomes largest
    int one=0, two=1, three=2;
    
    double max_distance=0;
    for(k=0; k < (int)polygon.size()-1; k++) {
        for(l=k+1; l < polygon.size(); l++) {
            cv::Point3d point = polygon[k]-polygon[l];
            double distance = point.dot(point);
            if(distance > max_distance) {
                max_distance=distance;
                one=k; two=l;
            }
        }
    }
    double max_length_crossproduct=0.0;
    for(k=0; k < polygon.size(); k++) {
        if( (k!=one) && (k!=two) ) {
            
            //Compute cross product
            cv::Point3d vec1 = polygon[two]-polygon[one];
            cv::Point3d vec2 = polygon[k]-polygon[one];
            cv::Point3d cross = vec1.cross(vec2);
            
            double lae=cross.dot(cross);
            
            if(lae>max_length_crossproduct) {
                max_length_crossproduct=lae;
                three=k;
            }
        }
    }
    
    if(max_length_crossproduct==0.0) {
        return normal;
    }
     
    cv::Point3d vec1 = polygon[two]-polygon[one];
    cv::Point3d vec2 = polygon[three]-polygon[one];
    cv::Point3d cross = vec1.cross(vec2);
    double lae=cross.dot(cross);
    if(lae>0) {
        normal=cross/sqrt(lae);
        
        //Check the orientation by computing the sign of the area size
        //0.5* normal \cdot \sum_{i=0}^n v_i \times v_{i+1}
        cv::Point3d vec;
        vec=polygon[polygon.size()-1].cross(polygon[0]);
        for(int i=0; i < polygon.size()-1; i++)
            vec=vec+polygon[i].cross(polygon[i+1]);
        double twice_area_size = normal.dot(vec);
        if(twice_area_size < 0.0) {
            normal = -1.0 * normal;
        }
        return(normal);
    } else return(normal); //no valid normal vector
}

bool equal2D(cv::Point2d vertex1, cv::Point2d vertex2) {
    cv::Point2d diff = vertex1 - vertex2;
    if(diff.dot(diff) < 0.01 * 0.01) return true;
    return false;
}
bool equal3D(cv::Point3d vertex1, cv::Point3d vertex2) {
    cv::Point3d diff = vertex1 - vertex2;
    if(diff.dot(diff) < 0.01 * 0.01) return true;
    return false;
}

unsigned int addVertex2D(cv::Point2d vertex, std::vector<vertex_2D_t>& vertices2D) {
    unsigned int i=0;
    for(i=0; i< vertices2D.size(); i++) {
        if(equal2D(vertex, vertices2D[i].point)) return i;
    }
    struct vertex_2D_t newvertex;
    newvertex.point=vertex;
    vertices2D.push_back(newvertex);
    return i;
}
unsigned int addVertex3D(cv::Point3d vertex, std::vector<cv::Point3d >& vertices3D) {
    unsigned int i=0;
    for(i=0; i< vertices3D.size(); i++) {
        if(equal3D(vertex, vertices3D[i])) return i;
    }
    vertices3D.push_back(vertex);
    return i;
}

//Compute the construction points of a corner
bool computeCorner(cv::Point2d A, cv::Point2d B, cv::Point2d C, double distanceAB, double distanceBC, cv::Point3d normalAB, cv::Point3d normalBC, cv::Point2d projector_left, cv::Point2d projector_right, std::vector<overhang_2Dvertex_t >& constructionPointsleftPolygon, std::vector<overhang_2Dvertex_t >& constructionPointsrightPolygon) {
        
    cv::Point2d a = B-A;
    double la=a.dot(a);
    cv::Point2d b = C-B;
    double lb=b.dot(b);
    if((la<=0)||(lb<=0)) {
        return false;
    }
    a/=sqrt(la); b/=sqrt(lb);
    
    cv::Point2d d1, d2;
    d1.x=a.y; d1.y=-a.x; d2.x=b.y; d2.y=-b.x;
    d1=distanceAB*d1;
    d2=distanceBC*d2;
    
    if((projector_left.x==0)&&(projector_left.y==0)) projector_left = d1;
    if((projector_right.x==0)&&(projector_right.y==0)) projector_right = d2;
       
    
    cv::Point2d P1,P2,P3;
    double s1,s4,s6,s8;
    
    //Compute P1
    //solve d1+s1 a = d2+s2 b <=> a s1 -b s2 = d2 - d1

    double detM = a.x*(-b.y) +b.x*a.y;
    if(fabs(detM)<0.0001) {
        //no intersection point, handle separately
        return false;
    }
    s1=((d2.x-d1.x)*(-b.y)+b.x*(d2.y-d1.y))/detM;
    //s2=(a.x*(d2.y-d1.y)-a.y*(d2.x-d1.x))/detM;
    
    P1=d1+s1*a;
    
    //GBB new:
    //If angle is too sharp then the intersection point might be too far away: Use projection instead
    if(sqrt(P1.dot(P1))>distanceAB+distanceBC) return false;
    
    if(equal3D(normalAB, normalBC)) {
        overhang_2Dvertex_t point;
        point.type = INTERSECTION_VERTEX;
        point.point = B+P1;
        constructionPointsleftPolygon.push_back(point);
        constructionPointsrightPolygon.push_back(point);
        return true;
    }
    
    //return false;
    
    //Compute ridge line direction vector
    
    cv::Point3d ridge = normalAB.cross(normalBC);
    cv::Point2d r;
    r.x=ridge.x; r.y=ridge.y;
    
    //Computer intersection of ridge line with both lines
    cv::Point2d P2a, P2b;
    //s3 r = P1 + s4 a <=> s3 r - s4 a = P1
    
    s4=0; s6=0;
    bool ok1=false;
    bool ok2=false;
    
    detM = r.x*(-a.y) - r.y*(-a.x);
    if(fabs(detM)>0.0001) {
        s4 = (r.x * P1.y - r.y * P1.x)/detM;
        P2a = P1 + s4 * a;
        if(sqrt(P2a.dot(P2a))<=distanceAB+distanceBC)
            ok1=true;
    }
                           
    
    //s5 r = P1 + s6 b <=> s5 r - s6 b = P1
    detM = r.x*(-b.y) - r.y*(-b.x);
    if(fabs(detM)>0.0001) {
        s6 = (r.x * P1.y - r.y * P1.x)/detM;
        P2b = P1 + s6 * b;
        if(sqrt(P2b.dot(P2b))<=distanceAB+distanceBC)
            ok2=true;
    }
    
    if((ok1 && ok2) ==false) {
        if((ok1 || ok2) == false) {
            overhang_2Dvertex_t point;
            point.type = INTERSECTION_VERTEX;
            point.point = B+P1;
            
            constructionPointsleftPolygon.push_back(point);
            constructionPointsrightPolygon.push_back(point);
            return true;
        } else if(ok1) {
            P2=P2a;
        } else
            P2=P2b;
    } else {
        if((s6<0)&&(s4<0)) {
            P2=P2a;
            ok2=false;
        } else {
            P2=P2b;
            ok1=false;
        }
    }
    //compute P3
    P3=P2;
    if(ok1) {
        //P2+s7 projector_right = P1 + s8 b <=> s7 projector_right - s8 b = P1-P2
        detM = projector_right.x * (-b.y) - projector_right.y * (-b.x);
        if(fabs(detM)>0.0001) {
            s8=(projector_right.x*(P1.y-P2.y)-projector_right.y*(P1.x-P2.x))/detM;
            P3=P1+s8*b;
        }
        
    } else {
        //P2+s7 projector_left = P1 + s8 a <=> s7 projector_left - s8 a = P1-P2
        detM = projector_left.x * (-a.y) - projector_left.y * (-a.x);
        if(fabs(detM)>0.0001) {
            s8=(projector_left.x*(P1.y-P2.y)-projector_left.y*(P1.x-P2.x))/detM;
            P3=P1+s8*a;
        }
    }
    
    if(ok1) {
        overhang_2Dvertex_t point;
        point.type = INTERSECTION_VERTEX;
        point.point = B+P2;
        
        constructionPointsleftPolygon.push_back(point);
        constructionPointsrightPolygon.push_back(point);
        
        point.type = PROJECTION_VERTEX;
        point.point = B+P3;
        
        constructionPointsrightPolygon.push_back(point);
    } else {
        overhang_2Dvertex_t point;
        point.type = PROJECTION_VERTEX;
        point.point = B+P3;
        
        constructionPointsleftPolygon.push_back(point);
        
        point.type = INTERSECTION_VERTEX;
        point.point = B+P2;
        constructionPointsleftPolygon.push_back(point);
        constructionPointsrightPolygon.push_back(point);
    }
    
    return true;
}

//Add a z-coordinate to a 2D vector
cv::Point3d make3DPoint(cv::Point2d point2d, cv::Point3d normal, cv::Point3d pointOnPlane) {
    cv::Point3d result;
    
    //Use Hesse normal form.
    result.x=point2d.x;
    result.y=point2d.y;
    result.z = (pointOnPlane.dot(normal)-point2d.x*normal.x - point2d.y*normal.y)/normal.z;
    
    return result;
}

//compute a plane equation for a plane on which a given polygon lies
 void computePlane(std::vector< cv::Point3d >& vertices_3D, std::vector<unsigned long>& wallpoints,cv::Point3d& centerpoint,  cv::Point3d& direction1,  cv::Point3d& direction2, cv::Point3d& normal) {
      
     //Compute normal
     std::vector<cv::Point3d> polygon;
     for(unsigned long i=0; i<wallpoints.size(); i++) {
         polygon.push_back(vertices_3D[wallpoints[i]]);
     }
     
     normal=computeNormal(polygon);
    
     //Compute center of gravity of vertices and longest edge
     cv::Point3d center(0,0,0);
     
     cv::Point3d longestDirection;
     cv::Point3d orthogonalDirection;
     double length=0;
     int anz = -1+(int)wallpoints.size();
     for(int i=0; i<anz; i++) {
         cv::Point3d  direction;
         center = center + vertices_3D[wallpoints[i]];
         direction = vertices_3D[wallpoints[i+1]] - vertices_3D[wallpoints[i]];
         double l=direction.dot(direction);
         if(l>=length) {
             longestDirection = direction;
             length=l;
         }
     }
     center=center/(double)anz;
     if(length>0) longestDirection=longestDirection/sqrt(length);
     //compute direction orthogonal to normal and to longest direction
     orthogonalDirection=normal.cross(longestDirection);
     length=orthogonalDirection.dot(orthogonalDirection);
     if(length>0) orthogonalDirection=orthogonalDirection/sqrt(length);
     
     centerpoint=center;
     direction1=longestDirection;
     direction2=orthogonalDirection;
 }

 //compute 2D coordinates in the 3D plane of a polygon
 void polygonTo2DPlane(std::vector< cv::Point3d >& vertices_3D, std::vector<unsigned long>& wallpoints,cv::Point3d& centerpoint,  cv::Point3d& direction1,  cv::Point3d& direction2, std::vector< cv::Point2d >& coordinates2D) {

     for(unsigned long k=0; k < wallpoints.size(); k++) {
         
         //compute local 2D coordinates
         cv::Point3d v = vertices_3D[wallpoints[k]]-centerpoint;
         
         //coordinates with respect to the basis:
         
         cv::Point2d point;
         point.x= v.dot(direction1);
         point.y= v.dot(direction2);
         
         coordinates2D.push_back(point);
     }
 }

//Transformation of 2D coordinates such that match1 is mapped to attach_point1 and match2 is mapped to attach_point2
bool computeImageCoordinates(std::vector< cv::Point2d >& coordinates2D, std::vector< cv::Point2d >& image_coordinates, cv::Point2d attach_point1, cv::Point2d attach_point2, cv::Point2d match1, cv::Point2d match2, double &minx, double &miny, double &maxx, double &maxy) {
    
    double l=sqrt( (match2.x-match1.x)*(match2.x-match1.x) + (match2.y-match1.y)*(match2.y-match1.y));
    if(l<0.001) {
        return false;
    }
    //scaling factor
    double r= sqrt((attach_point2.x-attach_point1.x)*(attach_point2.x-attach_point1.x)+(attach_point2.y-attach_point1.y)*(attach_point2.y-attach_point1.y))/l;
    
    //perform shift
    cv::Point3d shift;
    shift.x= attach_point1.x - r*match1.x;
    shift.y= attach_point1.y - r*match1.y;
    
    match2.x=r*match2.x+shift.x;
    match2.y=r*match2.y+shift.y;
    //compute rotation angle alpha
    double alpha_attach = atan2(attach_point2.y-attach_point1.y, attach_point2.x - attach_point1.x );
    double alpha_match  = atan2(match2.y-attach_point1.y, match2.x - attach_point1.x );
    double alpha = alpha_attach - alpha_match;
    
    for(int i=0; i<coordinates2D.size(); i++) {
        cv::Point2d point;
        point.x = r*coordinates2D[i].x+shift.x - attach_point1.x;
        point.y = r*coordinates2D[i].y+shift.y - attach_point1.y;
        
        //rotate:
        double x = cos(alpha) * point.x - sin(alpha) * point.y + attach_point1.x;
        double y = sin(alpha) * point.x + cos(alpha) * point.y + attach_point1.y;
        
        point.x = x;
        point.y = y;
        
        if(x<minx) minx=x;
        if(x>maxx) maxx=x;
        if(y<miny) miny=y;
        if(y>maxy) maxy=y;
        
        image_coordinates.push_back(point);
    }
    
    
    return true;
}


int cmpInt(const void *a, const void *b) {
    int x = *(int *) a;
    int y = *(int *) b;
    
    return (x-y);
}

//generate a perspectively corrected facade texture image
void computeImage(std::vector<TVec2f>& texcoords, std::vector< cv::Point2d >& image_coordinates, double minx, double miny, cv::Mat& textureImage, cv::Mat& walltexture) {
    
    cv::Point2f corner[4]; //4 points in the texture image
    cv::Point2f dest[4];  //4 points in the layout image
         
    //start at the left side and look for further vertices with a maximal value of the minimum of distances to the previously selected vertices
    int index[5]={0,1,2,3,0};
    
    int numberCoordinates = (int)image_coordinates.size();

    if(numberCoordinates >4) {
             
         for(int k=1; k<-1+(int)image_coordinates.size(); k++ ) {
             if(image_coordinates[k].x < image_coordinates[index[0]].x)
                 index[0]=k;
         }
         
         for(int kk=1; kk< 3; kk++) {
             double maxdistance=-1;
             
             for(int kkk=0; kkk < -1+image_coordinates.size(); kkk++) {
                 //the index must not be selected before
                 bool used=false;
                 for(int l=0; (l<kk)&& !used; l++) {
                     if(index[l]==kkk) used=true;
                 }
                 if(used) continue;
                 
                 cv::Point2d diff = image_coordinates[index[0]]-image_coordinates[kkk];
                 double mindist=diff.dot(diff);
                 
                 for(int ii=1; ii<kk; ii++) {
                     diff=image_coordinates[index[ii]]-image_coordinates[kkk];
                     double d=diff.dot(diff);
                     if(d<mindist) mindist=d;
                 }
                 
                 if(maxdistance<mindist) {
                     maxdistance=mindist;
                     index[kk]=kkk;
                 }
             }
         }
        
         //search fourth point that does not lie on the three straight lines through the other points
         double maxdet=-1;
      
         cv::Point2d r[3];
         r[0]=image_coordinates[index[1]]-image_coordinates[index[0]];
         r[1]=image_coordinates[index[2]]-image_coordinates[index[0]];
         r[2]=image_coordinates[index[2]]-image_coordinates[index[1]];
         
         for(int kkk=0; kkk < -1+(int)image_coordinates.size(); kkk++) {
             //the index must not be selected before
             bool used=false;
             for(int l=0; (l<3)&& !used; l++) {
                 if(index[l]==kkk) used=true;
             }
             if(used) continue;
             
             cv::Point2d s[3];
             s[0]=image_coordinates[kkk]-image_coordinates[index[0]];
             s[1]=s[0];
             s[2]=image_coordinates[kkk]-image_coordinates[index[1]];
             
             double mindet=-1;
             for(int j=0;j<3;j++) {
                 double det=fabs(s[j].x*r[j].y-s[j].y*r[j].x);
                 if(mindet==-1) mindet=det;
                 else if(det<mindet) mindet=det;
             }
             if(mindet > maxdet) {
                 maxdet=mindet;
                 index[3]=kkk;
             }
         }
         if(maxdet<=0.01) {
             //std::cout << "Warning: Problem finding fourth point for texture with " << numberCoordinates << " points" << std::endl;
             numberCoordinates=3; //no additional point found
         }
              
    }
     if(numberCoordinates > 4) numberCoordinates = 4;
     else numberCoordinates = 3;
     
     //sort the index
     
     if(numberCoordinates ==4) {
         qsort(index,4,sizeof(int),cmpInt);
         index[4]=index[0];
     } else {
         qsort(index,3,sizeof(int),cmpInt);
         index[3]=index[0];
     }
         
     for(int k=0; k<numberCoordinates; k++) {
         corner[k].x= texcoords[index[k]].x;
         corner[k].y= texcoords[index[k]].y;
         
         dest[k].x=image_coordinates[index[k]].x-minx;
         dest[k].y=image_coordinates[index[k]].y-miny;
     }
     

     for(int k=0; k<numberCoordinates; k++) {
         corner[k].x=corner[k].x*(double)textureImage.cols;
         corner[k].y=corner[k].y*(double)textureImage.rows;
     }
     if(numberCoordinates >= 4) {
         cv::Mat pt=cv::getPerspectiveTransform(corner,dest);
         cv::warpPerspective(textureImage, walltexture, pt, walltexture.size());
     }
     else {
         cv::Mat at=cv::getAffineTransform(corner,dest);
         cv::warpAffine(textureImage, walltexture, at, walltexture.size());
     }
}

struct match_t {
    unsigned long edge;
    unsigned long start_index_polygon;
    unsigned long end_index_polygon;
};

#define RESOLUTION 15.0 //15 Pixel pro Meter

//Estimate the size of overhangs based on an edge image of the facade texture
double estimateWithEdges(cv::Mat& truncatedImage) {
    
    if(truncatedImage.rows<3) return 0;
    
    //cv::flip(truncatedImage,truncatedImage,0);
    //cv::imshow("Truncated image",truncatedImage);
    
    
    cv::Mat greyImage, cannyImage;
    cvtColor(truncatedImage, greyImage, CV_8U);
    //Gauss-Filter
    //cv::GaussianBlur( greyImage,  greyImage, cv::Size(3, 3), 2.0);

    cv::Canny(greyImage, cannyImage, 50, 200, 3 );
    //cv::imshow("Edges",cannyImage);
    
    double histogrammCanny[truncatedImage.rows/2];
    int maxCanny=0, minCanny=255*truncatedImage.cols;
    for(int i=0; i<truncatedImage.rows/2; i++) {
        histogrammCanny[i]=0;
        for(int j=0; j<truncatedImage.cols; j++) {
            histogrammCanny[i]+=cannyImage.at<unsigned char>(i,j);
        }
        if(histogrammCanny[i] > maxCanny) maxCanny = histogrammCanny[i];
        if(histogrammCanny[i] < minCanny) minCanny = histogrammCanny[i];
    }
    //norming
    double factorCanny = maxCanny-minCanny+1;

    for(int i=0; i<truncatedImage.rows/2; i++) {
        histogrammCanny[i] = (histogrammCanny[i]-minCanny)*255 / factorCanny;
    }
    
    //Determine position of first canny row with entry above 200.
    double h=0;
    for(int i=0; i<truncatedImage.rows/2; i++) {
        if(histogrammCanny[i]>200) {
            h=((double)i)/RESOLUTION;
            break;
            
        }
    }
    return h;
}

//Estimate the size of an overhang based on a line separating the dominant colors of the roof and the facade
double estimateWithColors(cv::Mat& truncatedImage, float roof_r, float roof_g, float roof_b, float wall_r, float wall_g, float wall_b) {
    
    if(truncatedImage.rows<3) return 0;
    
    //cv::flip(truncatedImage,truncatedImage,0);
    //cv::imshow("Truncated image",truncatedImage);
    
    double histogramm[truncatedImage.rows/2];
    double maxHist=0, minHist=100000;
    
    
    for(int i=0; i<truncatedImage.rows/2; i++) {
        histogramm[i]=0;
        for(int j=0; j<truncatedImage.cols; j++) {
            
            cv::Vec3b temp=truncatedImage.at<cv::Vec3b>(i,j);
            float b = ((float)temp[0])/255.0;
            float g = ((float)temp[1])/255.0;
            float r = ((float)temp[2])/255.0;
            
            double dist_roof = sqrt((r-roof_r)*(r-roof_r)+(g-roof_g)*(g-roof_g)+(b-roof_b)*(b-roof_b));
            double dist_wall = sqrt((r-wall_r)*(r-wall_r)+(g-wall_g)*(g-wall_g)+(b-wall_b)*(b-wall_b));
            
            if(dist_roof < 0.25*dist_wall) histogramm[i]++;
            
        }
        if(histogramm[i] > maxHist) maxHist = histogramm[i];
        if(histogramm[i] < minHist) minHist = histogramm[i];
    }
    //norming
    double factor = maxHist-minHist;
    if(factor == 0) factor=1;

    for(int i=0; i<truncatedImage.rows/2; i++) {
        histogramm[i]= (histogramm[i]-minHist) / factor;
    }
    
    //Find the line index that separates roof and facade by approximation a rectangular function
    //Compute distance to zero function:
    double error=0;
    for(int i=0; i<truncatedImage.rows/2; i++) {
        error+=histogramm[i]*histogramm[i];
    }
    //Find best rectangle function to approximate the histogram in l2 norm
    int bestindex = 0;
    double smallest_error = error;
    for(int i=0; i<truncatedImage.rows/2; i++) {
        
        //error-=histogramm[i]*histogramm[i];
        //error+=(1-histogramm[i])*(1-histogramm[i]);
        error += 1-2*histogramm[i];
        
        if(error < smallest_error) {
            smallest_error = error;
            bestindex = i;
        }
    }
    
    //Compute size in meter from bestindex
    
    double h=0;
    h=((double)bestindex)/RESOLUTION;
    return h;
}

//Determine the color of a roof facet from its texture
bool findDominantTextureColor(std::vector<TVec2f> &texcoords, std::string &texturename, float &red, float &green, float &blue ) {
    static std::string lastImage="";
    static cv::Mat textureImage;

    if(lastImage!=texturename) {
         lastImage=texturename;
         
         std::stringstream name;
         std::string buffer;
         name.str(std::string());
         name << path_input << texturename;
         buffer = name.str();
             
         //std::cout << "Versuche zu lesen: " << buffer << std::endl;
         //buffer = buffer.replace(buffer.length()-4,4,".png");
         
         textureImage=cv::imread(buffer);
         cv::flip(textureImage,textureImage,0);

         if((textureImage.rows>0)&&(textureImage.cols>0)) {
           //cv::imshow("Read texture",textureImage);
         }
     }
     if((textureImage.rows==0) || (textureImage.cols==0)) {
         return false; //no texture available
     }
                         
     //Perspectice correction not required, just use texture coordinates
     //compute mask of facet region
     cv::Mat mask = cv::Mat(textureImage.rows,textureImage.cols, CV_8UC1, cv::Scalar(0));
       
     std::vector<cv::Point> rand;
     cv::Point pt;
     for(int i=0; i < texcoords.size(); i++) {
           
         pt.x= (int)(((double)textureImage.cols)*texcoords[i].x);
         pt.y= (int)(((double)textureImage.rows)*texcoords[i].y);
         if(pt.x>=textureImage.cols) pt.x=textureImage.cols-1;
         if(pt.y>=textureImage.rows) pt.y=textureImage.rows-1;
           
         rand.push_back(pt);
     }
       
     const cv::Point* ppt[1] = { &rand[0] };
       
     int npt[] = { (int)rand.size() };
     cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255), 8);
     
     return mean_color(textureImage, mask, red, green, blue);
}

//Determine the color of a facade from its texture
bool findDominantTextureColor(std::vector<TVec2f> &texcoords, cv::Mat textureImage, float &red, float &green, float &blue ) {
                         
     //Perspectice correction not required, just use texture coordinates
     //compute mask of facet region
     cv::Mat mask = cv::Mat(textureImage.rows,textureImage.cols, CV_8UC1, cv::Scalar(0));
       
     std::vector<cv::Point> rand;
     cv::Point pt;
     for(int i=0; i < texcoords.size(); i++) {
           
         pt.x= (int)(((double)textureImage.cols)*texcoords[i].x);
         pt.y= (int)(((double)textureImage.rows)*texcoords[i].y);
         if(pt.x>=textureImage.cols) pt.x=textureImage.cols-1;
         if(pt.y>=textureImage.rows) pt.y=textureImage.rows-1;
           
         rand.push_back(pt);
     }
       
     const cv::Point* ppt[1] = { &rand[0] };
       
     int npt[] = { (int)rand.size() };
     cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255), 8);
    
     //cv::imshow("Maske",mask);
     //cv::imshow("Texture",textureImage);
         
     return mean_color(textureImage, mask, red, green, blue);
}


void analyzeTexture(const citygml::CityObject& building, std::vector< cv::Point3d >& vertices_3D,
std::vector<vertex_2D_t>& vertices_2D, std::vector<edge_t>& edges, std::vector<polygon_t>& polygons) {
    
    std::string lastImage="";
    cv::Mat textureImage;
    
    int anzChilds = building.getChildCityObjectsCount();
    
    for(int p=0; p<anzChilds; p++) {
        const citygml::CityObject &buildingElement = building.getChildCityObject(p);
        std::string objtype = buildingElement.getTypeAsString();
        
        
        if(objtype != "WallSurface") continue; //process only (textured) walls
        
        //read coordinate list
        for(unsigned int i=0; i<buildingElement.getGeometriesCount(); i++) {
            
            int anzPolygons = buildingElement.getGeometry(i).getPolygonsCount();
            for(unsigned int k=0; k<anzPolygons; k++) {
                
                std::shared_ptr<const citygml::Polygon> citypolygon = buildingElement.getGeometry(i).getPolygon(k);
                std::shared_ptr<const citygml::LinearRing> ring= citypolygon->exteriorRing();
                std::vector<TVec3d> ringvertices=ring->getVertices();
                
                //look for associated texture coordinates
                std::vector<TVec2f> texcoords;
                
                std::vector<std::string> themes = citypolygon->getAllTextureThemes(true);
                if(themes.size()>0) {
                    texcoords = citypolygon->getTexCoordsForTheme(themes[0], true);
                    
                    //if there are interior polygons, the number of texture coordinates might exceed the number of outer vertices
                    while(texcoords.size()>ringvertices.size()) {
                        texcoords.pop_back();
                    }
                    if(texcoords.size()==ringvertices.size()) {
                        
                        //Texture exists. Find all roof edges that belong to this texture
                        std::vector< unsigned long > ind3D;
                        std::vector< struct match_t > correspondingEdges;
                        
                        for(unsigned long j=0;j<ringvertices.size();j++) {
                            cv::Point3d vertex3d;
                     
                            vertex3d.x=ringvertices[j].x;
                            vertex3d.y=ringvertices[j].y;
                            vertex3d.z=ringvertices[j].z;
                 
                            ind3D.push_back(addVertex3D(vertex3d, vertices_3D));
                        }
                            
                        if(ind3D.size()>=3) {
                            
                            //If first point is not repeated: add it as last point
                            if(ind3D[ind3D.size()-1]!=ind3D[0]) {
                                ind3D.push_back(ind3D[0]);
                            }
                            
                            //Jetzt die passenden Roof edges finden
                            for(unsigned long kk=0; kk<edges.size(); kk++) {
                                
                                for(unsigned long ii=0; ii<ind3D.size()-1; ii++) {
                                    if((edges[kk].relevant) && (!edges[kk].inside)) {
                                        if((edges[kk].end_vertex_3D == ind3D[ii]) && (edges[kk].start_vertex_3D == ind3D[ii+1])) {
                                            struct match_t match;
                                            match.edge=kk;
                                            match.start_index_polygon=ii;
                                            match.end_index_polygon=ii+1;
                                            
                                            //Match: Texture belongs to a roof edge
                                            correspondingEdges.push_back(match);
                                        }
                                    }
                                }
                            }
                            
                            if(correspondingEdges.size()>0) {
                                //The texture belongs to at least one roof edge
                            
                                std::shared_ptr<const citygml::Texture> texture = citypolygon->getTextureFor(themes[0]);
                                std::string texturename= texture->getUrl();
                                
                                if(lastImage!=texturename) {
                                    lastImage=texturename;
                                    
                                    std::stringstream name;
                                    std::string buffer;
                                    name.str(std::string());
                                    name << path_input << texturename;
                                    buffer = name.str();
                                        
                                    //std::cout << "Versuche zu lesen: " << buffer << std::endl;
                                    //buffer = buffer.replace(buffer.length()-4,4,".png");
                                    
                                    textureImage=cv::imread(buffer);
                                    cv::flip(textureImage,textureImage,0);
   
                                    if((textureImage.rows>0)&&(textureImage.cols>0)) {
                                      //cv::imshow("Read texture",textureImage);
                                    }
                                }
                                if((textureImage.rows==0) || (textureImage.cols==0)) {
                                    continue; //no texture available
                                }
                                
                                //Determine dominant color of this facade wall
                                float r,g,b;
                                
                                bool dominantColorExists=findDominantTextureColor(texcoords, textureImage, r, g, b);
                                
                                //Map wall polygon onto the wall plane (2D coordinates)
                                std::vector< cv::Point2d > coordinates2D;
                                //To this end: Compute plane parameters of the polygon
                                cv::Point3d centerpoint, normal;
                                cv::Point3d direction1, direction2;
                                computePlane(vertices_3D, ind3D, centerpoint, direction1, direction2, normal);
                                polygonTo2DPlane(vertices_3D, ind3D, centerpoint, direction1, direction2,  coordinates2D);
                                                                
                                //Compute overhang size for all roof edges belonging to this texture
                                for(unsigned long j=0; j<correspondingEdges.size(); j++) {
                                    
                                    cv::Point3d v = vertices_3D[edges[correspondingEdges[j].edge].end_vertex_3D]-vertices_3D[edges[correspondingEdges[j].edge].start_vertex_3D];
                                    
                                    double l=sqrt(v.dot(v));
                                    if(l>0.01) {
                            
                                        //Transform 2D coordinates so that current edge is on top
                                        std::vector< cv::Point2d > image_coordinates;
                                        cv::Point2d attach_point1, attach_point2, match1, match2;
                                        attach_point1.x=attach_point1.y=0.0;
                                        attach_point2.x=l*RESOLUTION; attach_point2.y=0.0;
                                        match1=coordinates2D[correspondingEdges[j].end_index_polygon];
                                        match2=coordinates2D[correspondingEdges[j].start_index_polygon];
                                        
                                        double minx, miny, maxx, maxy;
                                        if(computeImageCoordinates(coordinates2D, image_coordinates, attach_point1, attach_point2, match1, match2, minx, miny, maxx, maxy)) {
                                            
                                            //generate the image to be analyzed
                                            cv::Mat analyzeImage(maxy-miny+1, maxx-minx+1, CV_8UC3, cv::Vec3b(255, 255, 255));
                                            computeImage(texcoords, image_coordinates, minx, miny, textureImage, analyzeImage);
                                            
                                            int links=image_coordinates[correspondingEdges[j].end_index_polygon].x-minx;
                                            int unten=image_coordinates[correspondingEdges[j].end_index_polygon].y-miny;
                                            int rechts=image_coordinates[correspondingEdges[j].start_index_polygon].x-minx;
                                            int oben=analyzeImage.rows - unten;
                                            
                                            if((rechts-links>0)&&(unten-oben>0)) {
                                                cv::Rect grid_rect(links, oben, rechts-links, unten-oben);
                                                cv::Mat truncatedImage = analyzeImage(grid_rect);
                                                
                                                //Estimate size of overhang-region in transformed texture image
                                                
                                                cv::flip(truncatedImage,truncatedImage,0);
                                                
                                                //A. based on edge image
                                                
                                                double h[2] ={0.0, 0.0};
                                                h[0] = estimateWithEdges(truncatedImage);
                                                
                                                //B. based on similar colors.
                                                
                                                
                                                unsigned long index = edges[correspondingEdges[j].edge].polygon;
                                                
                                                if(dominantColorExists && (polygons[index].color_set)) {
                                                
                                                    h[1] = estimateWithColors(truncatedImage, polygons[index].r, polygons[index].g, polygons[index].b, r, g, b);
                                                    edges[correspondingEdges[j].edge].from_color = true;
                                                    
                                                    //For testing: draw detected lines into the truncated image
                                                    /*
                                                    cv::Point p1,p2;
                                                    p1.x=0;p2.x=truncatedImage.cols-1;
                                                    p2.y=p1.y=(int)(h[0]*(double)RESOLUTION);
                                                    cv::line(truncatedImage,p1,p2,cv::Vec3b(255,0,0), 1,  8,0);
                                                    p1.x=0;p2.x=truncatedImage.cols-1;
                                                    p2.y=p1.y=(int)(h[1]*(double)RESOLUTION);
                                                    cv::line(truncatedImage,p1,p2,cv::Vec3b(0,0,255), 1,  8,0);
                                                    std::cout << h[0]*(double)RESOLUTION << "; " << h[1]*(double)RESOLUTION << std::endl;
                                                    cv::imshow("Detected lines", truncatedImage);
                                                    cv::Mat dominantColors(truncatedImage.rows, truncatedImage.cols, CV_8UC3, cv::Vec3b(255, 255, 255));
                                                    for(int kk=0; kk < dominantColors.rows; kk++) {
                                                        for(int ll=0; ll<dominantColors.cols; ll++) {
                                                            if(kk < (int)(h[1]*(double)RESOLUTION))
                                                                dominantColors.at<cv::Vec3b>(kk,ll)=cv::Vec3b(polygons[index].b*255.0,polygons[index].g*255.0,polygons[index].r*255.0);
                                                            else
                                                                dominantColors.at<cv::Vec3b>(kk,ll)=cv::Vec3b(b*255.0,g*255.0,r*255.0);
                                                        }
                                                    }
                                                    cv::imshow("Dominant Colors", dominantColors);
                                                    cv::waitKey(0);
                                                    */
                                                }
                                                
                                                double d[2];
                                                double tanalpha = polygons[index].normal.z/sqrt(polygons[index].normal.x*polygons[index].normal.x+polygons[index].normal.y*polygons[index].normal.y);
                                                
                                                for(int kk=0; kk<2; kk++) {
                                                    d[kk] = h[kk];
                                            
                                                    if( polygons[index].normal.z < 0.999) {
                                                      
                                                        d[kk]=h[kk] * tanalpha/(1+tanalpha);
                                                    }
                                                    
                                                    if(d[kk]>2.0) d[kk]=2.0; //Maximum size of overhangs: 2 m
                                                }
                                                edges[correspondingEdges[j].edge].size = d[0];
                                                edges[correspondingEdges[j].edge].size_from_color = d[1];
                                                edges[correspondingEdges[j].edge].from_texture = true;
                                                
                                            }
                                        }
                                    }
                                }
                                
                            }
                        
                        }
                        
                    } else std::cout << "Warning: Number " << texcoords.size() << " of texture coordinates does not fit with number of vertices " << ringvertices.size() << "." << std::endl;
                }
             
            }
        }
    }
}

//To analyze difference between inliers and outliers
std::vector<cv::Point2d > inliereOutlierDistance;

void analyzePCL(std::vector< cv::Point3d >& vertices_3D, std::vector<edge_t>& edges, std::vector<polygon_t>& polygons) {
    
    for(unsigned long i=0; i<edges.size(); i++) {
        if(edges[i].relevant) {
            
            //Determine relevant x-y-rectangle
            cv::Point2d vertex[5], direction;
            
            vertex[0].x = vertices_3D[edges[i].end_vertex_3D].x - pointcloud.offset_x;
            vertex[0].y = vertices_3D[edges[i].end_vertex_3D].y - pointcloud.offset_y;
            
            vertex[1].x = vertices_3D[edges[i].start_vertex_3D].x - pointcloud.offset_x;
            vertex[1].y = vertices_3D[edges[i].start_vertex_3D].y - pointcloud.offset_y;
            
            cv::Point2d outer_normal;
            outer_normal.x = vertex[0].y - vertex[1].y;
            outer_normal.y = - vertex[0].x + vertex[1].x;
            double d=outer_normal.dot(outer_normal);
            if(d<0.01) continue;
            d=sqrt(d);
            outer_normal/=d;
        
            //Only use the middle part of this edge to avoid interference
            direction = vertex[1] - vertex[0];
            vertex[0] += 0.25*direction;
            vertex[1] -= 0.25*direction;
            
            vertex[4] = vertex[0];
            
            
            double eckelinksx=vertex[0].x, eckelinksy=vertex[0].y, eckerechtsx=vertex[0].x, eckerechtsy=vertex[0].y;
            
            vertex[2] = vertex[1] + 2.0*outer_normal;
            vertex[3] = vertex[0] + 2.0*outer_normal;
            
            for(int j=1; j<4; j++) {
                if(vertex[j].x < eckelinksx) eckelinksx = vertex[j].x;
                if(vertex[j].y < eckelinksy) eckelinksy = vertex[j].y;
                if(vertex[j].x > eckerechtsx) eckerechtsx = vertex[j].x;
                if(vertex[j].y > eckerechtsy) eckerechtsy = vertex[j].y;
            }
            
            cv::Point3d plane_normal = polygons[edges[i].polygon].normal;
            if(plane_normal.z==0) continue;
        
            //Compute index area for the rectangle in the quadtree
            double breitex=(pointcloud.pxmax-pointcloud.pxmin)/(double)SUBDIVISIONS;
            double breitey=(pointcloud.pymax-pointcloud.pymin)/(double)SUBDIVISIONS;
            
            int startx,starty,endx,endy;
            if(eckelinksx<pointcloud.pxmin) continue; //Edge lies at the border of the point cloud, do not process
            else if(eckelinksx>=pointcloud.pxmax) continue; //Edge lies at the border of the point cloud, do not process
            else startx=(int)((eckelinksx-pointcloud.pxmin)/breitex);
            if(eckerechtsx<pointcloud.pxmin) continue; //Edge lies at the border of the point cloud, do not process
            else if(eckerechtsx>=pointcloud.pxmax) continue; //Edge lies at the border of the point cloud, do not process
            else endx=(int)((eckerechtsx-pointcloud.pxmin)/breitex);
            if(eckelinksy<pointcloud.pymin) continue; //Edge lies at the border of the point cloud, do not process
            else if(eckelinksy>=pointcloud.pymax) continue; //Edge lies at the border of the point cloud, do not process
            else starty=(int)((eckelinksy-pointcloud.pymin)/breitey);
            if(eckerechtsy<pointcloud.pymin) continue; //Edge lies at the border of the point cloud, do not process
            else if(eckerechtsy>=pointcloud.pymax) continue; //Edge lies at the border of the point cloud, do not process
            else endy=(int)((eckerechtsy-pointcloud.pymin)/breitey);
            
            //Find nearest point that deviates largely from plane, find farest point that fits with plane and compute
            //arithmetic mean.
            
            double nearest_outlier=2.0, farest_inlier=0.0;
            bool inlierFound = false;
            bool outlierFound = false;
            
            for(int xx=startx;xx<=endx;xx++) {
                for(int yy=starty;yy<=endy;yy++) {
                    //iterate through the points belonging to the corresponding segment
                    for(long k=pointcloud.pktindex[xx][yy].start;k<=pointcloud.pktindex[xx][yy].ende;k++) {
                        
                        cv::Point2d point;
                        point.x=pointcloud.pointcloud[k].x;
                        point.y=pointcloud.pointcloud[k].y;
                        
                        //Check if the point lies within the rectangle in 2D
                        bool inside = true;
                        
                        double distance=0.0;
                        
                        for(int j=0; (j<4) && inside; j++) {
                            //outer normal of rectangle edge
                            cv::Point2d normal;
                            normal.x=vertex[j+1].y-vertex[j].y;
                            normal.y=-(vertex[j+1].x-vertex[j].x);
                            
                            if( point.dot(normal) > vertex[j].dot(normal) ) {
                                inside = false;
                            } else if(j==0) {
                                double len = normal.dot(normal);
                                if(len>0.0001) {
                                    normal/=sqrt(len);
                                    distance=fabs(point.dot(normal) - vertex[j].dot(normal));
                                } else inside = false;
                            }
                        }
                        
                        if(inside) {
                            //Check distance to plane in 3D
                            cv::Point3d planePoint;
                            planePoint.x = vertices_3D[edges[i].end_vertex_3D].x - pointcloud.offset_x;
                            planePoint.y = vertices_3D[edges[i].end_vertex_3D].y - pointcloud.offset_y;
                            planePoint.z = vertices_3D[edges[i].end_vertex_3D].z;
                            
                            double diff = pointcloud.pointcloud[k].dot(plane_normal) - planePoint.dot(plane_normal);
                            
                            if(fabs(diff)<=1.0) {
                                //Distance to extended roof plane is at most 1 m
                                //distance to roof border: distance
                                if(farest_inlier < distance) farest_inlier = distance;
                                inlierFound = true;
                            } else {
                                
                                double z = (planePoint.dot(plane_normal) - pointcloud.pointcloud[k].x * plane_normal.x -
                                pointcloud.pointcloud[k].y * plane_normal.y)/plane_normal.z;
                                
                                double z0 = pointcloud.pointcloud[k].z;
                                if(z0 <= z) { //Only if point is lower than plane
                                    
                                    if(nearest_outlier > distance) nearest_outlier = distance;
                                    outlierFound = true;
                                }
                            }
                        }
                        
                    }
                }
            }

            edges[i].from_cloud = true;
            if(inlierFound) {
                if(outlierFound)
                    edges[i].size_from_cloud = (nearest_outlier + farest_inlier)/2.0;
                else
                    edges[i].size_from_cloud = farest_inlier;
                
                //For evaluation only
                if(outlierFound) {
                    cv::Point2d inloutl;
                    inloutl.x = farest_inlier;
                    inloutl.y = nearest_outlier;
                    inliereOutlierDistance.push_back(inloutl);
                }
                
            } else {
                edges[i].size_from_cloud = 0;
            }
            if(!edges[i].from_texture) edges[i].size = edges[i].size_from_cloud;
        }
    }
}

int cmpDouble(const void *a, const void *b) {
    double x = *(double *) a;
    double y = *(double *) b;
    
    return (int)((x-y)*10000.0);
}

// Cumulate all differences

static std::vector<double> diff_edges_cloud;
static std::vector<double> diff_color_cloud;
static std::vector<double> diff_color_edges;

void addStatistik(std::vector<edge_t>& edges) {
    for(unsigned long i=0; i<edges.size(); i++) {
      if(edges[i].relevant) {
          if(edges[i].from_texture && edges[i].from_cloud) {
              diff_edges_cloud.push_back(edges[i].size-edges[i].size_from_cloud);
          }
          if(edges[i].from_color && edges[i].from_cloud) {
              diff_color_cloud.push_back(edges[i].size_from_color-edges[i].size_from_cloud);
          }
          if(edges[i].from_color) {
              diff_color_edges.push_back(edges[i].size_from_color-edges[i].size);
          }
      }
    }
}

void statistik() {
    if(diff_edges_cloud.size()>0) {
        std::cout << "Comparison beetween sizes from edge images and sizes from point cloud" << std::endl;
        qsort(&diff_edges_cloud[0],diff_edges_cloud.size(),sizeof(double),cmpDouble);
        std::cout << "Comparison for " << diff_edges_cloud.size() << " edges: texture (edge) size - point cloud size." << std::endl;
        std::cout << "Median: " << diff_edges_cloud[diff_edges_cloud.size()/2] << std::endl;
        std::cout << "0.25 Quartile: " << diff_edges_cloud[diff_edges_cloud.size()/4] << std::endl;
        std::cout << "0.75 Quartile: " << diff_edges_cloud[(diff_edges_cloud.size()*3)/4] << std::endl;
        std::cout << "Range: " << diff_edges_cloud[0] << " to " << diff_edges_cloud[diff_edges_cloud.size()-1] << std::endl;
        double mean=0;
        for(int i=0; i<diff_edges_cloud.size(); i++) mean+=diff_edges_cloud[i];
        mean/=(double)(diff_edges_cloud.size());
        std::cout << "Arithmetic mean: " << mean << std::endl << std::endl;
    }
    if(diff_color_cloud.size()>0) {
        std::cout << "Comparison beetween sizes from color comparison and sizes from point cloud" << std::endl;
        qsort(&diff_color_cloud[0],diff_color_cloud.size(),sizeof(double),cmpDouble);
        std::cout << "Comparison for " << diff_color_cloud.size() << " edges: color size - point cloud size." << std::endl;
        std::cout << "Median: " << diff_color_cloud[diff_color_cloud.size()/2] << std::endl;
        std::cout << "0.25 Quartile: " << diff_color_cloud[diff_color_cloud.size()/4] << std::endl;
        std::cout << "0.75 Quartile: " << diff_color_cloud[(diff_color_cloud.size()*3)/4] << std::endl;
        std::cout << "Range: " << diff_color_cloud[0] << " to " << diff_color_cloud[diff_color_cloud.size()-1] << std::endl;
        double mean=0;
        for(int i=0; i<diff_color_cloud.size(); i++) mean+=diff_color_cloud[i];
        mean/=(double)(diff_color_cloud.size());
        std::cout << "Arithmetic mean: " << mean << std::endl << std::endl;
    }
    if(diff_color_edges.size()>0) {
        std::cout << "Comparison: color comparison minus size from edges" << std::endl;
        qsort(&diff_color_edges[0],diff_color_edges.size(),sizeof(double),cmpDouble);
        std::cout << "Comparison for " << diff_color_edges.size() << " edges: color size - edge size." << std::endl;
        std::cout << "Median: " << diff_color_edges[diff_color_edges.size()/2] << std::endl;
        std::cout << "0.25 Quartile: " << diff_color_edges[diff_color_edges.size()/4] << std::endl;
        std::cout << "0.75 Quartile: " << diff_color_edges[(diff_color_edges.size()*3)/4] << std::endl;
        std::cout << "Range: " << diff_color_edges[0] << " to " << diff_color_edges[diff_color_edges.size()-1] << std::endl;
        double mean=0;
        for(int i=0; i<diff_color_edges.size(); i++) mean+=diff_color_edges[i];
        mean/=(double)(diff_color_edges.size());
        std::cout << "Arithmetic mean: " << mean << std::endl;
    }
    
    if(inliereOutlierDistance.size()>0) {
        std::vector<double> dist;
        for(unsigned long i=0; i<inliereOutlierDistance.size(); i++) {
            dist.push_back(inliereOutlierDistance[i].y-inliereOutlierDistance[i].x);
            //Outlier - Inlier
        }
        qsort(&dist[0],dist.size(),sizeof(double),cmpDouble);
        std::cout << "Comparison for " << dist.size() << " pairs of inlier and outlier distances" << std::endl;
        std::cout << "Median of difference nearest outlier - farest inlier distance: " << dist[dist.size()/2] << std::endl;
        std::cout << "0.25 Quartile: " << dist[dist.size()/4] << std::endl;
        std::cout << "0.75 Quartile: " << dist[dist.size()*3/4] << std::endl;
        std::cout << "Range: " << dist[0] << " to " << dist[dist.size()-1] << std::endl;
        double mean=0;
        for(int i=0; i<dist.size(); i++) mean+=dist[i];
        mean/=(double)(dist.size());
        std::cout << "Arithmetic mean: " << mean << std::endl;
    }
    
}


void processBuilding(const citygml::CityObject& building, std::vector< struct overhangs_of_roof_facet_t >& overhangs, std::vector< struct footprint_edge_t >& footprints) {
    
    std::vector< cv::Point3d > vertices_3D;
    std::vector<vertex_2D_t> vertices_2D;
    std::vector<polygon_t> polygons;
    std::vector<edge_t> edges;
    
    int anzChilds = building.getChildCityObjectsCount();
    
    for(int p=0; p<anzChilds; p++) {
        const citygml::CityObject &buildingElement = building.getChildCityObject(p);
        std::string objtype = buildingElement.getTypeAsString();
        
        
        if(objtype != "RoofSurface") continue; //process only roofs
        
        //read coordinate list
        for(unsigned int i=0; i<buildingElement.getGeometriesCount(); i++) {
            
            int anzPolygons = buildingElement.getGeometry(i).getPolygonsCount();
            unsigned int LoD = buildingElement.getGeometry(i).getLOD();
            
            for(unsigned int k=0; k<anzPolygons; k++) {
                
                std::vector<cv::Point3d> polygonPoints;
                struct polygon_t polygon;
                
                polygon.color_set = false;
                
                
                std::shared_ptr<const citygml::Polygon> citypolygon = buildingElement.getGeometry(i).getPolygon(k);
                std::shared_ptr<const citygml::LinearRing> ring= citypolygon->exteriorRing();
                std::vector<TVec3d> ringvertices=ring->getVertices();
                
                
                
                std::vector< unsigned long > ind2D;
                std::vector< unsigned long > ind3D;
                
                for(unsigned long j=0;j<ringvertices.size();j++) {
                    cv::Point3d vertex3d;
                    cv::Point2d vertex2d;
                    vertex2d.x=vertex3d.x=ringvertices[j].x;
                    vertex2d.y=vertex3d.y=ringvertices[j].y;
                    vertex3d.z=ringvertices[j].z;
                    
                    polygonPoints.push_back(vertex3d);
                    
                    ind2D.push_back(addVertex2D(vertex2d, vertices_2D));
                    ind3D.push_back(addVertex3D(vertex3d, vertices_3D));
                }
                    
                if(polygonPoints.size()>=3) {
                    
                    //If first point is not repeated: add it as last point
                    if(ind3D[ind3D.size()-1]!=ind3D[0]) {
                        ind3D.push_back(ind3D[0]);
                        ind2D.push_back(ind2D[0]);
                        polygonPoints.push_back(polygonPoints[0]);
                    }
                    
                    polygon.normal=computeNormal(polygonPoints);
                    
                    if(polygon.normal.dot(polygon.normal)>0) {
                        //Outer roof polygon has been read
                        //Create edges of polygon and link them with 2D vertices
                        for(int j=0; j<polygonPoints.size()-1; j++) {
                            struct edge_t edge;
                            edge.start_vertex_2D=ind2D[j];
                            edge.end_vertex_2D=ind2D[j+1];
                            edge.start_vertex_3D=ind3D[j];
                            edge.end_vertex_3D=ind3D[j+1];
                            edge.polygon=polygons.size();
                            edge.relevant=true;
                            edge.inside=false;
                            edge.size=0.4;
                            edge.size_from_cloud=0;
                            edge.LoD=LoD;
                            edge.from_cloud = edge.from_texture = edge.from_color = false;

                            polygon.edges.push_back(edges.size());
                            vertices_2D[edge.start_vertex_2D].departing_edges.push_back(edges.size());
                            vertices_2D[edge.end_vertex_2D].arriving_edges.push_back(edges.size());
                            edges.push_back(edge);
                        }
                        
                        //Process inner polygons
                        const std::vector< std::shared_ptr<citygml::LinearRing> > interiorrings = citypolygon->interiorRings();
                        for(unsigned int ii=0; ii< interiorrings.size(); ii++) {
                            std::vector<TVec3d> ringvertices=interiorrings[ii]->getVertices();
                            
                            std::vector< unsigned long > ind2D;
                            std::vector< unsigned long > ind3D;
                            
                            for(unsigned long j=0;j<ringvertices.size();j++) {
                                cv::Point3d vertex3d;
                                cv::Point2d vertex2d;
                                vertex2d.x=vertex3d.x=ringvertices[j].x;
                                vertex2d.y=vertex3d.y=ringvertices[j].y;
                                vertex3d.z=ringvertices[j].z;
                                
                                ind2D.push_back(addVertex2D(vertex2d, vertices_2D));
                                ind3D.push_back(addVertex3D(vertex3d, vertices_3D));
                            }
                            if(ind2D.size()>3) {
                                
                                //If first point is not repeated: add it as last point
                                if(ind3D[ind3D.size()-1]!=ind3D[0]) {
                                    ind3D.push_back(ind3D[0]);
                                    ind2D.push_back(ind2D[0]);
                                }
                                
                                for(int j=0; j<ind2D.size()-1; j++) {
                                    struct edge_t edge;
                                    edge.start_vertex_2D=ind2D[j];
                                    edge.end_vertex_2D=ind2D[j+1];
                                    edge.start_vertex_3D=ind3D[j];
                                    edge.end_vertex_3D=ind3D[j+1];
                                    edge.polygon=polygons.size();
                                    edge.relevant=true;
                                    edge.inside=false;
                                    edge.size = 0.4;
                                    edge.size_from_cloud = 0;
                                    edge.LoD=LoD;
                                    edge.from_cloud = edge.from_texture = edge.from_color = false;
                                    
                                    polygon.edges.push_back(edges.size());
                                    vertices_2D[edge.start_vertex_2D].departing_edges.push_back(edges.size());
                                    vertices_2D[edge.end_vertex_2D].arriving_edges.push_back(edges.size());
                                    edges.push_back(edge);
                                }
                            }
                        }
                        
                        //Read texture of roof polygon and determine the dominant color.
                        std::vector<std::string> themes = citypolygon->getAllTextureThemes(true);
                        if(themes.size()>0) {
                            std::vector<TVec2f> texcoords = citypolygon->getTexCoordsForTheme(themes[0], true);
                            
                            //if there are interior polygons, the number of texture coordinates might exceed the number of outer vertices
                            while(texcoords.size()>ringvertices.size()) {
                                texcoords.pop_back();
                            }
                            if(texcoords.size()==ringvertices.size()) {
                                std::shared_ptr<const citygml::Texture> texture = citypolygon->getTextureFor(themes[0]);
                                std::string texturename= texture->getUrl();

                                polygon.color_set = findDominantTextureColor(texcoords, texturename, polygon.r, polygon.g, polygon.b);
                            }
                        }
                        
                        
                        polygons.push_back(polygon);
                    }
                }
            }
        }
    }

    //Mark all edges as irrelevant if they are traversed in both directions in 3D or if they
    //are traversed in opposite direction at a higher altitude
    
    for(unsigned long i=0; i<edges.size(); i++) {
        for(unsigned long j=0; j<vertices_2D[edges[i].end_vertex_2D].departing_edges.size(); j++) {
            struct edge_t opposite_edge = edges[vertices_2D[edges[i].end_vertex_2D].departing_edges[j]];
            if(opposite_edge.end_vertex_2D==edges[i].start_vertex_2D) {
                edges[i].inside=true;
                edges[i].size=0.2;
                //edge in opposite direction found, check z-coordinates
                if((vertices_3D[edges[i].start_vertex_3D].z < vertices_3D[opposite_edge.end_vertex_3D].z+0.2) ||
                   (vertices_3D[edges[i].end_vertex_3D].z < vertices_3D[opposite_edge.start_vertex_3D].z+0.2)) {
                    edges[i].relevant=false;
                }
            }
        }
        if(edges[i].relevant) {
            //Check if vertex is adjacent to a footprint edge of a different building(part)
            cv::Point2d start = vertices_2D[edges[i].start_vertex_2D].point;
            cv::Point2d end = vertices_2D[edges[i].end_vertex_2D].point;
            cv::Point2d verdir = end-start;
            double l=sqrt(verdir.dot(verdir));
            if(l<0.001) edges[i].relevant = false;
            else {
                verdir/=l;
                for(unsigned long j=0; j<footprints.size(); j++) {
                    //vectors have to point into the same direction because normal of footprint points down
                    if(footprints[j].direction.dot(verdir)>0.99999) {
                        if(fabs( (start.x-footprints[j].left.x)*verdir.y-(start.y-footprints[j].left.y)*verdir.x ) < 0.4) {
                            //lines do match and point into the same direction
                            //check, if edges really overlap
                            //footprints[j].left + r footprint vector = start + t (verdir.y, -verdir.x)
                            //<=>  r footprint vector + t  (-verdir.y, +verdir.x) = start-footprints[j].left;
                            
                            double detM = (footprints[j].right.x-footprints[j].left.x)*verdir.x - (footprints[j].right.y-footprints[j].left.y) * (-verdir.y);
                            if(fabs(detM)>0.00001) {
                                double r = ( (start.x-footprints[j].left.x)*verdir.x - (start.y-footprints[j].left.y) * (-verdir.y) )/detM;
                                
                                //footprints[j].left + s footprint vector = end + t (verdir.y, -verdir.x)
                                //<=>  s footprint vector + t  (-verdir.y, +verdir.x) = end-footprints[j].left;
                                double s = ( (end.x-footprints[j].left.x)*verdir.x - (end.y-footprints[j].left.y) * (-verdir.y) )/detM;
                                
                                if((s>0.001)&&(r<0.999)) {
                                    if( (footprints[j].left_z > vertices_3D[ edges[i].start_vertex_3D ].z-0.3 ) ||
                                       (footprints[j].left_z > vertices_3D[ edges[i].end_vertex_3D ].z-0.3 ) ||
                                       (footprints[j].right_z > vertices_3D[ edges[i].start_vertex_3D ].z-0.3 ) ||
                                       (footprints[j].right_z > vertices_3D[ edges[i].end_vertex_3D ].z-0.3 ) ) {
                                        edges[i].relevant = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    //Find facade textures for relevant roof edges
    analyzeTexture(building, vertices_3D, vertices_2D, edges, polygons);
    
    //if PCL: Add PCL distance
    if(pcl_filename!="") {
        analyzePCL(vertices_3D, edges, polygons);
        //For evaluation:
        addStatistik(edges);
    }
    
    //For each polygon: Add overhangs to relevant edges
    for(unsigned long i=0; i<polygons.size(); i++) {
        if(polygons[i].normal.z<0.0001) {
            continue;
        }
        
        struct overhangs_of_roof_facet_t overhangs_of_one_polygon;
        overhangs_of_one_polygon.color_set = polygons[i].color_set;
        overhangs_of_one_polygon.r = polygons[i].r;
        overhangs_of_one_polygon.g = polygons[i].g;
        overhangs_of_one_polygon.b = polygons[i].b;
        for(unsigned long j=0; j<polygons[i].edges.size(); j++) {
            
            if(edges[polygons[i].edges[j]].relevant) {
                
                //The size of an overhang is larger if it is outside the footprint
                double size = edges[polygons[i].edges[j]].size;
                
                long arriving_edge_ind=-1;
                long departing_edge_ind=-1;
                
                struct edge_t current = edges[polygons[i].edges[j]];

                cv::Point2d direction = vertices_2D[current.end_vertex_2D].point-vertices_2D[current.start_vertex_2D].point;
                double l=direction.dot(direction);
                if(l<=0.001*0.001) {
                    continue; //Length of edge is zero
                    
                }
                direction/=sqrt(l);
                
                //Check if there exists exactly one previous edge
                //Also compute a vector that is used instead of orthogonal projection if orthogonal projection would lead to
                //an intersection
                
                cv::Point2d left_projector;
                left_projector.x = direction.y;
                left_projector.y = -direction.x;
                double left_projector_cos_angle =0;
           
                
                for(unsigned k=0; k<vertices_2D[current.start_vertex_2D].arriving_edges.size(); k++) {
                    if(edges[vertices_2D[current.start_vertex_2D].arriving_edges[k]].relevant) {
                        if(edges[vertices_2D[current.start_vertex_2D].arriving_edges[k]].end_vertex_3D == current.start_vertex_3D) {
                            if(arriving_edge_ind==-1) {
                                arriving_edge_ind = vertices_2D[current.start_vertex_2D].arriving_edges[k];
                            } else arriving_edge_ind = -2; //multiple arriving and matching edges
                        }
                    }
                    if(vertices_3D[edges[vertices_2D[current.start_vertex_2D].arriving_edges[k]].end_vertex_3D].z >= vertices_3D[current.start_vertex_3D].z-0.2) {
                        
                        //This is a vertex that might intersect with a constructed extension
                        cv::Point2d dirpred = vertices_2D[edges[vertices_2D[current.start_vertex_2D].arriving_edges[k]].end_vertex_2D].point - vertices_2D[edges[vertices_2D[current.start_vertex_2D].arriving_edges[k]].start_vertex_2D].point;
                        double l=dirpred.dot(dirpred);
                        if(l>0) {
                            dirpred=-dirpred/sqrt(l);
                            double cos_angle = dirpred.dot(direction);
                            if((cos_angle > left_projector_cos_angle)&&(cos_angle<1)&&(direction.y*dirpred.x-direction.x*dirpred.y >0)) {
                                //angle smaller than 90 degrees
                               
                                    //edge is on the problematic side
                                    left_projector_cos_angle = cos_angle;
                                    left_projector = dirpred;
                                
                                
                            }
                        }
                        
                    }
                }
                if(arriving_edge_ind>=0) {
                    if(edges[arriving_edge_ind].relevant==false) arriving_edge_ind=-1;
                }
                
                //Check if there exists exactly one following edge
                //Also compute a direction for projection without intersection
                double right_projector_cos_angle = 0;
                cv::Point2d right_projector;
                right_projector.x = direction.y;
                right_projector.y = -direction.x;
                
                for(unsigned k=0; k<vertices_2D[current.end_vertex_2D].departing_edges.size(); k++) {
                    if(edges[vertices_2D[current.end_vertex_2D].departing_edges[k]].relevant) {
                        if(edges[vertices_2D[current.end_vertex_2D].departing_edges[k]].start_vertex_3D == current.end_vertex_3D) {
                            if(departing_edge_ind==-1) {
                                departing_edge_ind = vertices_2D[current.end_vertex_2D].departing_edges[k];
                            } else departing_edge_ind = -2; //multiple arriving and matching edges
                        }
                    }
                    
                    if(vertices_3D[edges[vertices_2D[current.end_vertex_2D].departing_edges[k]].start_vertex_3D].z >= vertices_3D[current.end_vertex_3D].z-0.2) {
                        
                        //This is a vertex that might intersect with a constructed extension
                        cv::Point2d dirsuc = vertices_2D[edges[vertices_2D[current.end_vertex_2D].departing_edges[k]].end_vertex_2D].point - vertices_2D[edges[vertices_2D[current.end_vertex_2D].departing_edges[k]].start_vertex_2D].point;
                        double l=dirsuc.dot(dirsuc);
                        if(l>0) {
                            dirsuc= dirsuc/sqrt(l); //+ instead of -
                            double cos_angle = dirsuc.dot(-direction);
                            if((cos_angle > right_projector_cos_angle)&&(cos_angle<1)&&(direction.y*dirsuc.x-direction.x*dirsuc.y >0)) {
                                //angle smaller than 90 degrees
                                cv::Point3d first, second;
                               
                                    //edge is on the problematic side
                                    right_projector_cos_angle = cos_angle;
                                    right_projector = dirsuc;
                                
                            }
                        }
                        
                    }
                }
                if(departing_edge_ind>=0) {
                    if(edges[departing_edge_ind].relevant==false) departing_edge_ind=-1;
                }
                                                
                //Construct left border of overhang
                std::vector< overhang_2Dvertex_t > constructionPointsleft, constructionPointsright;
                
                double arriving_size = 0.4, departing_size = 0.4;
                cv::Point2d dummy_projector(0,0);
                
                if(arriving_edge_ind<0) {
                    
                    //orthogonal (or non-orthogonal if intersection) extension
                    cv::Point2d corner = vertices_2D[current.start_vertex_2D].point + size/sqrt(1-left_projector_cos_angle*left_projector_cos_angle) * left_projector;
                    overhang_2Dvertex_t point;
                    point.point = corner;
                    point.type = PROJECTION_VERTEX;
                    cv::Point2d distance = corner-vertices_2D[current.start_vertex_2D].point;
                    if(sqrt(distance.dot(distance))> 3*size) continue;
                    
                    constructionPointsleft.push_back(point);
                } else {
                    arriving_size = edges[arriving_edge_ind].size;
                    std::vector< overhang_2Dvertex_t > constructionPointsOtherPolygon;
                    
                    bool erg=computeCorner(vertices_2D[edges[arriving_edge_ind].start_vertex_2D].point,vertices_2D[edges[arriving_edge_ind].end_vertex_2D].point,vertices_2D[current.end_vertex_2D].point, arriving_size, size, polygons[edges[arriving_edge_ind].polygon].normal,polygons[i].normal, dummy_projector, left_projector, constructionPointsOtherPolygon, constructionPointsleft);
                    
                    if(!erg) {
                        cv::Point2d corner = vertices_2D[current.start_vertex_2D].point + size/sqrt(1-left_projector_cos_angle*left_projector_cos_angle) * left_projector;
                        overhang_2Dvertex_t point;
                        point.point = corner;
                        point.type = PROJECTION_VERTEX;
                        cv::Point2d distance = corner-vertices_2D[current.start_vertex_2D].point;
                        if(sqrt(distance.dot(distance))> 3*size) continue;
                        constructionPointsleft.push_back(point);
                    }
                }
                
                //Construct right border of overhang
                if(departing_edge_ind<0) {
                    //orthogonal extension
                    cv::Point2d corner = vertices_2D[current.end_vertex_2D].point + size/sqrt(1-right_projector_cos_angle*right_projector_cos_angle) * right_projector;
                    overhang_2Dvertex_t point;
                    point.point = corner;
                    point.type = PROJECTION_VERTEX;
                    cv::Point2d distance = corner-vertices_2D[current.end_vertex_2D].point;
                    if(sqrt(distance.dot(distance))> 3*size) continue;
                    constructionPointsright.push_back(point);
                } else {
                    
                    departing_size = edges[departing_edge_ind].size;
                    std::vector< overhang_2Dvertex_t > constructionPointsOtherPolygon;
                    
                    bool erg =
                    computeCorner(vertices_2D[current.start_vertex_2D].point, vertices_2D[current.end_vertex_2D].point,
                                  vertices_2D[edges[departing_edge_ind].end_vertex_2D].point,  size, departing_size,
                                  polygons[i].normal,polygons[edges[departing_edge_ind].polygon].normal, right_projector, dummy_projector, constructionPointsright, constructionPointsOtherPolygon);
                    
                    if(!erg) {
                        cv::Point2d corner = vertices_2D[current.end_vertex_2D].point + size/sqrt(1-right_projector_cos_angle*right_projector_cos_angle) * right_projector;
                        overhang_2Dvertex_t point;
                        point.point = corner;
                        point.type = PROJECTION_VERTEX;
                        cv::Point2d distance = corner-vertices_2D[current.end_vertex_2D].point;
                        if(sqrt(distance.dot(distance))> 3*size) continue;
                        constructionPointsright.push_back(point);
                    }
                }
                
                //Assemble overhang polygon if last left construction points does not lie on the right side of the first right
                //construction point
                cv::Point2d left = constructionPointsleft[constructionPointsleft.size()-1].point;
                cv::Point2d right = constructionPointsright[0].point;
                //left + r*direction = right
                double r;
                if(fabs(direction.x)>fabs(direction.y))
                     r = (right.x-left.x)/direction.x;
                else
                    r = (right.y-left.y)/direction.y;
                if(r<=0) {
                    continue; //wrong order of points
                }
                
                std::vector<struct overhang_vertex_t> overhang;
                struct overhang_vertex_t vertex;
                vertex.type = WALL_VERTEX;
                vertex.LoD = edges[polygons[i].edges[j]].LoD;
                vertex.point = vertices_3D[current.start_vertex_3D];
                overhang.push_back(vertex);
                for(unsigned int k=0; k < constructionPointsleft.size(); k++) {
                    vertex.point = make3DPoint(constructionPointsleft[k].point,polygons[i].normal,vertices_3D[current.start_vertex_3D]);
                    vertex.type = constructionPointsleft[k].type;
                    overhang.push_back(vertex);
                }
                for(unsigned int k=0; k < constructionPointsright.size(); k++) {
                    vertex.point = make3DPoint(constructionPointsright[k].point,polygons[i].normal,vertices_3D[current.start_vertex_3D]);
                    vertex.type = constructionPointsright[k].type;
                    overhang.push_back(vertex);
                }
                vertex.type = WALL_VERTEX;
                vertex.point = vertices_3D[current.end_vertex_3D];
                overhang.push_back(vertex);
                vertex.point = vertices_3D[current.start_vertex_3D];
                overhang.push_back(vertex);
                
                
                overhangs_of_one_polygon.points.push_back(overhang);
            }
            
        }
        if(overhangs_of_one_polygon.points.size()>0)
            overhangs.push_back(overhangs_of_one_polygon);
    }
}

long find(const char* str1, const char* str2, long start) {
    long i=start;
    while(str1[i]!=0) {
        const char* s2=str2;
        long j=i;
        while((str1[j]==*s2)&&(*s2!=0)) {
            j++; s2++;
        }
        if(*s2==0) /*string found*/
            return(j);
        i++;
    }
    return(i);
}

void writePolygon(FILE*file, std::vector< cv::Point3d >& points, std::string &id, int &counter) {
    
    fprintf(file,"<gml:surfaceMember><gml:Polygon gml:id=\"%s_O%d\"><gml:exterior><gml:LinearRing>",id.c_str(),counter++);
            fprintf(file,"<gml:posList srsDimension=\"3\">");
    
    for(unsigned int k =0; k< points.size(); k++) {
        
        fprintf(file,"%.3f %.3f %.3f",points[k].x, points[k].y, points[k].z);
        if(k<points.size()-1) fprintf(file," ");
    }
    fprintf(file,"</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>");
}


//Merge neighboring roof vertices, but avoid inner polygons.
void simplify(std::vector< std::vector<overhang_vertex_t> >&  points) {
    
    if(points.size() <= 1) return;
    
    for(unsigned long i=0; i<points.size()-1; i++) {
        unsigned long nexti=i+1;
        if(i==points.size()-1) nexti=0;
        bool found=false;
        std::vector<overhang_vertex_t> merged;
        for(unsigned long j=0; j<points[i].size()-1; j++) {
            if( ((points[i][j].type == WALL_VERTEX) && (points[i][j+1].type == INTERSECTION_VERTEX)) ||
               ((points[i][j+1].type == WALL_VERTEX) && (points[i][j].type == INTERSECTION_VERTEX)) ) {
                //intersection edge found, lokk if there is a match with the next polygon
                
                unsigned long k=0;
                for(k=0; k<points[nexti].size()-1; k++) {
                    if( ((points[nexti][k].type == WALL_VERTEX) && (points[nexti][k+1].type == INTERSECTION_VERTEX)) ||
                           ((points[nexti][k+1].type == WALL_VERTEX) && (points[nexti][k].type == INTERSECTION_VERTEX)) ) {
                            //intersection edge of next polygon found
                        if( equal3D(points[nexti][k].point, points[i][j+1].point) && equal3D(points[nexti][k+1].point, points[i][j].point)) {
                            //Matching edge: Join polygons here!
                            found=true;
                            
                            break;
                        }
                    }
                }
                
                if(found) {
                    
                    for(unsigned long l=0; l<=j; l++) {
                        merged.push_back(points[i][l]);
                        
                    }
                    
                    for(unsigned long l=k+2; l<points[nexti].size()-1; l++) {
                        merged.push_back(points[nexti][l]);
                    }
                    
                    for(unsigned long l=0; l<k; l++) {
                        merged.push_back(points[nexti][l]);
                    }
                    
                    for(unsigned long l=j+1; l < points[i].size(); l++) {
                        merged.push_back(points[i][l]);
                        
                    }
                    break;
                }
                
            }
        }
        
        if(found) {
            //Replace two polygons by the new polygon merged
            
            if(nexti>i) {
                points[i] = merged;
                for(unsigned long k=i+1; k<points.size()-1; k++)
                    points[k]=points[k+1];
                points.pop_back();
                i--;
                if(points.size()<=1) break;
            } else {
                //Merged with the first polygon
                points[0] = merged;
                points.pop_back();
                break;
            }
        }
    }
}

void update_CityGML_file(std::vector< struct buildingOverhangs_t > &overhangs_all_buildings ) {
    
    std::vector< struct farbe_t> farbe;
    
    //Read GML data from file with name input_filename and insert overhangs
    std::ifstream ifs;
    ifs.open(input_filename.c_str(), std::ifstream::in);
    if (!ifs.is_open()) {
        std::cout << "Could not open input file." << std::endl;
        return;
    }
    ifs.seekg(0, ifs.end);
    long length = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char * inputGML = new char[length];
    ifs.read(inputGML, length);
    ifs.close();
    
    long inputPos=0;
    
    //Open output file
    std::stringstream name;
    std::string buffer;
    name.str(std::string());
    if(output_filename == "") {
        name << path_input << "model_with_overhangs.gml";
    } else {
        name << output_filename;
    }
    buffer = name.str();
    
    FILE* file = fopen(buffer.c_str(),"wb");
    if(!file){
       std::cout << "Could not write output file: " << buffer << std::endl;
       fclose(file);
       return;
    }
    
    for(unsigned long i=0; i<overhangs_all_buildings.size(); i++) {
        
        //Counter for new polygons
        int counter=0;
        
        if(overhangs_all_buildings[i].roof_facet.size()==0) continue;
        char searchstring[200];
        std::string id;
        if(overhangs_all_buildings[i].type==BUILDING) {
            id = overhangs_all_buildings[i].building_id;
            sprintf(searchstring,"<bldg:Building gml:id=\"%s\">",id.c_str());
        }
        else {
            id = overhangs_all_buildings[i].buildingPart_id;
            sprintf(searchstring,"<bldg:BuildingPart gml:id=\"%s\">",id.c_str());
        }
        
        long insertionPoint = find(inputGML, searchstring, inputPos);
        
        insertionPoint = find(inputGML, "<bldg:boundedBy>", insertionPoint);
        
        if( inputGML[insertionPoint] == 0) {
            std::cout << "Error, different order!" << std::endl;
        }
        insertionPoint-=16;
        
        char merke=inputGML[insertionPoint];
        inputGML[insertionPoint]=0;
        fprintf(file, "%s", inputGML+inputPos);
        inputGML[insertionPoint]=merke;
        inputPos=insertionPoint;
        
        //fprintf(file,"<bldg:BuildingInstallation>\n");
        
        //iterate through roof polygons of this building
        for(unsigned long j=0; j< overhangs_all_buildings[i].roof_facet.size(); j++) {
            
            if(overhangs_all_buildings[i].roof_facet[j].points.size()==0) continue;
            int LoD = 2;
            if(overhangs_all_buildings[i].roof_facet[j].points[0].size()>0) {
                LoD = overhangs_all_buildings[i].roof_facet[j].points[0][0].LoD;
            }
            
            //------------------------
            struct farbe_t einefarbe;
            einefarbe.r = overhangs_all_buildings[i].roof_facet[j].r;
            einefarbe.g = overhangs_all_buildings[i].roof_facet[j].g;
            einefarbe.b = overhangs_all_buildings[i].roof_facet[j].b;
            int start_counter = counter;
            
            //Simplify overhangs for this roof polygon
            //Merge roof (and ceiling) polygons
            std::vector< std::vector<overhang_vertex_t> > merged = overhangs_all_buildings[i].roof_facet[j].points;
            simplify(merged);
            
            fprintf(file,"<bldg:boundedBy><bldg:RoofSurface>\n");
            fprintf(file,"<bldg:lod%dMultiSurface><gml:MultiSurface>\n",LoD);
            
            //iterate through all overhang polygons of this roof polygon
            for(unsigned long jj=0; jj< overhangs_all_buildings[i].roof_facet[j].points.size(); jj++) {
            
                if(merged[jj].size()<4) continue;
                
                std::vector<cv::Point3d> polygon;
                for(long k=0; k<merged[jj].size(); k++) {
                    polygon.push_back(merged[jj][k].point);
                }
                
                writePolygon(file, polygon, id, counter);
            }
            
            fprintf(file,"</gml:MultiSurface></bldg:lod%dMultiSurface></bldg:RoofSurface></bldg:boundedBy>",LoD);
            
            
            fprintf(file,"<bldg:boundedBy><bldg:RoofSurface>\n");
            fprintf(file,"<bldg:lod%dMultiSurface><gml:MultiSurface>\n",LoD);
            
            //iterate through all overhang polygons of this roof polygon to add ceiling
            for(unsigned long jj=0; jj< merged.size(); jj++) {
            
                if(merged[jj].size()<4) continue;
                
                std::vector<cv::Point3d> polygon;
                
                //add ceiling
                for(long k=merged[jj].size()-1; k>=0; k--) {
                    polygon.push_back(merged[jj][k].point);
                }
                for(unsigned long k=0; k< polygon.size(); k++) {
                    polygon[k].z-=0.3;
                }
                writePolygon(file, polygon, id, counter);
            }
            
            fprintf(file,"</gml:MultiSurface></bldg:lod%dMultiSurface></bldg:RoofSurface></bldg:boundedBy>",LoD);
            
            
            //iterate through all overhang polygons of this roof polygon to add walls
            for(unsigned long jj=0; jj< overhangs_all_buildings[i].roof_facet[j].points.size(); jj++) {
            
                if(overhangs_all_buildings[i].roof_facet[j].points[jj].size()<4) continue;
                
                for(unsigned long k=0; k<overhangs_all_buildings[i].roof_facet[j].points[jj].size()-1; k++) {
                    
                    //Do not add walls between wall segments
                    if((overhangs_all_buildings[i].roof_facet[j].points[jj][k+1].type==WALL_VERTEX)&&(overhangs_all_buildings[i].roof_facet[j].points[jj][k].type==INTERSECTION_VERTEX)) continue;
                    if((overhangs_all_buildings[i].roof_facet[j].points[jj][k+1].type==INTERSECTION_VERTEX)&&(overhangs_all_buildings[i].roof_facet[j].points[jj][k].type==WALL_VERTEX)) continue;
                    
                    
                    std::vector<cv::Point3d> polygon;
                    cv::Point3d point1 = overhangs_all_buildings[i].roof_facet[j].points[jj][k+1].point;
                    cv::Point3d point2 = overhangs_all_buildings[i].roof_facet[j].points[jj][k].point;
                    polygon.push_back(point1); polygon.push_back(point2);
                    point1.z-=0.3; point2.z-=0.3;
                    polygon.push_back(point2); polygon.push_back(point1);
                    polygon.push_back(overhangs_all_buildings[i].roof_facet[j].points[jj][k+1].point);
                    
                    fprintf(file,"<bldg:boundedBy><bldg:RoofSurface>\n");
                    fprintf(file,"<bldg:lod%dMultiSurface><gml:MultiSurface>\n",LoD);
                    writePolygon(file, polygon, id, counter);
                    fprintf(file,"</gml:MultiSurface></bldg:lod%dMultiSurface></bldg:RoofSurface></bldg:boundedBy>",LoD);
                }
            }
            
            if(overhangs_all_buildings[i].roof_facet[j].color_set) {
                for(int k=start_counter; k<counter; k++) {
                    std::string pid = id;
                    pid.append("_O");
                    pid.append(std::to_string(k));
                    einefarbe.id.push_back(pid);
                }
                if(einefarbe.id.size()>0)
                    farbe.push_back(einefarbe);
            }
        }
        //fprintf(file,"</bldg:BuildingInstallation>");
        
    }
    //Add appearance members for overhang polygons
    long insertionPoint = find(inputGML, "</core:CityModel>", inputPos);
    if( inputGML[insertionPoint] != 0) {

        insertionPoint-=17;
        inputGML[insertionPoint]=0;
        fprintf(file, "%s", inputGML+inputPos);
        
        for(unsigned long i=0; i<farbe.size(); i++) {
            fprintf(file, "<app:appearanceMember><app:Appearance><app:surfaceDataMember><app:X3DMaterial>\n");
            fprintf(file, "<app:diffuseColor>%.3f %.3f %.3f</app:diffuseColor>\n",farbe[i].r, farbe[i].g, farbe[i].b);
            fprintf(file, "<app:transparency>0.000</app:transparency>\n");
            
            for(unsigned long j=0; j<farbe[i].id.size(); j++) {
                fprintf(file, "<app:target>#%s</app:target>\n",farbe[i].id[j].c_str());
            }
            
            fprintf(file, "</app:X3DMaterial></app:surfaceDataMember></app:Appearance></app:appearanceMember>\n");
        }
        fprintf(file, "</core:CityModel>");
    } else {
        fprintf(file, "%s", inputGML+inputPos);
    }
    
    fclose(file);
}


void processAllBuildings(std::shared_ptr<const citygml::CityModel> city, std::vector< struct footprint_edge_t >& footprints )
{
    std::vector< struct buildingOverhangs_t > overhangs_all_buildings;
    const citygml::ConstCityObjects& roots = city->getRootCityObjects();
    for ( unsigned int i = 0; i < roots.size(); i++ ) {
        const citygml::CityObject& object = *roots[i];
        
    
        std::string basistyp = object.getTypeAsString();
        if((basistyp!="Building")&&(basistyp!="BuildingPart")) continue;
    
        int anzChilds = object.getChildCityObjectsCount();
        if(anzChilds==0) continue;
        
        const citygml::CityObject &firstObj = object.getChildCityObject(0);
        std::string firstObjtype = firstObj.getTypeAsString();
        if(firstObjtype=="BuildingPart") {
            
            //Building consists of building parts
             for(int p=0; p<anzChilds; p++) {
                const citygml::CityObject &buildingPartObj = object.getChildCityObject(p);
                std::string objtype = buildingPartObj.getTypeAsString();
                if(objtype!="BuildingPart") continue;
                struct buildingOverhangs_t overhangs;
                overhangs.building_id = object.getId();
                overhangs.buildingPart_id = buildingPartObj.getId();
                overhangs.type=BUILDING_PART;
                 
                 processBuilding(buildingPartObj,overhangs.roof_facet,footprints);
                overhangs_all_buildings.push_back(overhangs);
            }
        }
        else {
            //no building parts
            struct buildingOverhangs_t overhangs;
            overhangs.building_id = object.getId();
            overhangs.type = BUILDING;
            processBuilding(object,overhangs.roof_facet,footprints);
            overhangs_all_buildings.push_back(overhangs);
        }
    }
    //update file
    std::cout << std::endl << "Write GML file" << std::endl;
    update_CityGML_file(overhangs_all_buildings);

}

void getEdges(const citygml::CityObject& building, std::vector< struct footprint_edge_t >& footprints) {
    
    std::vector< struct footprint_edge_t > localfootprints;
    
    int anzChilds = building.getChildCityObjectsCount();
    
    for(int p=0; p<anzChilds; p++) {
        const citygml::CityObject &buildingElement = building.getChildCityObject(p);
        std::string objtype = buildingElement.getTypeAsString();
        
        
        if(objtype != "GroundSurface") continue; //process only footprints
        
        //read coordinate list
        for(unsigned int i=0; i<buildingElement.getGeometriesCount(); i++) {
            
            int anzPolygons = buildingElement.getGeometry(i).getPolygonsCount();
            for(unsigned int k=0; k<anzPolygons; k++) {
                
                std::shared_ptr<const citygml::Polygon> citypolygon = buildingElement.getGeometry(i).getPolygon(k);
                std::shared_ptr<const citygml::LinearRing> ring= citypolygon->exteriorRing();
                std::vector<TVec3d> ringvertices=ring->getVertices();
     
                if(ringvertices.size()>2) {
                    struct footprint_edge_t before, after;
                    
                    before.left_z = -10000.0;
                    before.right_z = -10000.0;
                    
                    for(unsigned long j=0;j<ringvertices.size();j++) {
                        after.left.x=before.right.x=ringvertices[j].x;
                        after.left.y=before.right.y=ringvertices[j].y;
                        
                        if( j>0 ) {
                            before.direction=before.right-before.left;
                            double l=before.direction.dot(before.direction);
                            if(l>0) {
                                before.direction/=sqrt(l);
                                localfootprints.push_back(before);
                            }
                        }
                        
                        before=after;
                    }
                    if(ringvertices[ringvertices.size()-1]!=ringvertices[0]) {
                        before.right.x=ringvertices[0].x;
                        before.right.y=ringvertices[0].y;
                        before.direction=before.right-before.left;
                        double l=before.direction.dot(before.direction);
                        if(l>0) {
                            before.direction/=sqrt(l);
                            localfootprints.push_back(before);
                        }
                    }
                }
            }
        }
    }
    
    //Compute maximum z-coordinates
    for(int p=0; p<anzChilds; p++) {
        const citygml::CityObject &buildingElement = building.getChildCityObject(p);
        std::string objtype = buildingElement.getTypeAsString();
        
        
        if(objtype != "WallSurface") continue; //process only footprints
        
        //read coordinate list
        for(unsigned int i=0; i<buildingElement.getGeometriesCount(); i++) {
            
            int anzPolygons = buildingElement.getGeometry(i).getPolygonsCount();
            for(unsigned int k=0; k<anzPolygons; k++) {
                
                std::shared_ptr<const citygml::Polygon> citypolygon = buildingElement.getGeometry(i).getPolygon(k);
                std::shared_ptr<const citygml::LinearRing> ring= citypolygon->exteriorRing();
                std::vector<TVec3d> ringvertices=ring->getVertices();
     
                if(ringvertices.size()>2) {
                    
                    for(unsigned long j=0;j<ringvertices.size();j++) {
                        cv::Point2d wallpoint;
                        wallpoint.x = ringvertices[j].x;
                        wallpoint.y = ringvertices[j].y;
                        
                        for(unsigned long l=0; l<localfootprints.size(); l++) {
                            if(equal2D( localfootprints[l].left, wallpoint ) && (localfootprints[l].left_z < ringvertices[j].z))
                                localfootprints[l].left_z = ringvertices[j].z;
                            if(equal2D( localfootprints[l].right, wallpoint ) && (localfootprints[l].right_z < ringvertices[j].z))
                            localfootprints[l].right_z = ringvertices[j].z;
                        }
                    }
                }
            }
        }
    }
    
    for(unsigned long i=0; i<localfootprints.size(); i++) {
        if(localfootprints[i].left_z<0) {
            localfootprints[i].left_z=10000; //no corresponding wall point!
        }
        if(localfootprints[i].right_z<0) {
            localfootprints[i].right_z=10000; //no corresponding wall point!
        }
        footprints.push_back(localfootprints[i]);
    }
}

//Read all footprint edges
void determineFootprintEdges(std::shared_ptr<const citygml::CityModel> city, std::vector< struct footprint_edge_t >& footprints )
{
    const citygml::ConstCityObjects& roots = city->getRootCityObjects();
    for ( unsigned int i = 0; i < roots.size(); i++ ) {
        const citygml::CityObject& object = *roots[i];
    
        std::string basistyp = object.getTypeAsString();
        if((basistyp!="Building")&&(basistyp!="BuildingPart")) continue;
    
        int anzChilds = object.getChildCityObjectsCount();
        if(anzChilds==0) continue;
        
        const citygml::CityObject &firstObj = object.getChildCityObject(0);
        std::string firstObjtype = firstObj.getTypeAsString();
        if(firstObjtype=="BuildingPart") {
            
            //Building consists of building parts
            for(int p=0; p<anzChilds; p++) {
                const citygml::CityObject &buildingPartObj = object.getChildCityObject(p);
                std::string objtype = buildingPartObj.getTypeAsString();
                if(objtype!="BuildingPart") continue;
                getEdges(buildingPartObj, footprints);
            }
        }
        else {
            //no building parts
            std::vector< std::vector< cv::Point3d >> overhangs;
            getEdges(object, footprints);
        }
    }
}

void usage()
{
    std::cout << "Usage: overhangs <input filename> [<output filename>] [-options...]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -pcl <filename.las>   Read a point cloud in las format and obtain overhangs from the cloud" << std::endl;
    std::cout << "If the output filename is not specified, the output is written to model_with_overhangs.gml" << std::endl;
}

int main(int argc, const char * argv[]) {
    
    const char *z;
    
    if((argc<2)||(argc>5)) {
        //z=default_input_filename;
        usage();
        return 1;
    }
    //Process input file name
    z=argv[1];
    
    int lastSlash=-1;
    int i=0;
    while(*z!=0) {
        if((*z=='/')||(*z=='\\')) lastSlash=i;
        i++; z++;
    }
    if(lastSlash>=0) {
        i=0;
        z=argv[1];
        char path[lastSlash+2];
        char *zz=path;
        while(i<=lastSlash) {
            *zz++=*z++; i++;
        }
        *zz=0;
        path_input=path;
    } else path_input="";
    
    input_filename = argv[1];
    
    if((argc==3)||(argc==5)) {
        output_filename = argv[2];
    }
    city = readModel();
    if(!city) {
        std::cout << "Could not read and parse " << argv[1] << std::endl;
        return 0;
    }
    
    if(argc>=4) {
        //Read point cloud
        std::string keyword = argv[argc-2];
        if(keyword == "-pcl")
           pcl_filename = argv[argc-1];
        else {
            usage();
            return 1;
        }
    }
    try {
        if(pcl_filename!="") {
            if(pcl_filename.find(".las")<pcl_filename.length()) {
                std::cout << "Consider point cloud in las format." << std::endl;
               readLAS(pcl_filename.c_str(), pointcloud);
            } else {
               std::cout << "Consider point cloud in ASCII XYZ format." << std::endl;
               readXYZ(pcl_filename.c_str(), pointcloud);
            }
        }
        
        
        //Initialize footprint vector for later comparison
        std::vector< struct footprint_edge_t > footprints;
        determineFootprintEdges(city, footprints);
        //------------------------------------------------
        
        processAllBuildings(city, footprints);
        statistik();

    } catch(...) {
        std::cout << "An error occurred, please check parameters." << std::endl;
        return 0;
    }
    return 0;
}
