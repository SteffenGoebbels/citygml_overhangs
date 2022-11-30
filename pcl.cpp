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

#include "pcl.hpp"
#include <liblas/liblas.hpp>
#include <fstream> // std::ifstream
#include <iostream> // std::cout
#include <math.h>
#include <stdint.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


 bool cmpx(cv::Point3d& a , cv::Point3d& b) {

     return a.x < b.x;
 }

bool cmpy(cv::Point3d& a , cv::Point3d& b) {

    return a.y < b.y;
}

 //Divide the point cloud into SUBDIVISIONS X SUBDIVISIONS subregions
 //xmin etc. belong to coordinates minus offset_x
 void quadtree(cloud_t& pointcloud) {
     long xindex[SUBDIVISIONS+1];
     long i,j, pos;
     double breitex=(pointcloud.pxmax-pointcloud.pxmin)/(double)SUBDIVISIONS;
     double breitey=(pointcloud.pymax-pointcloud.pymin)/(double)SUBDIVISIONS;
     cv::Point3d *p;
     
     p=&pointcloud.pointcloud[0];
     sort(pointcloud.pointcloud.begin(), pointcloud.pointcloud.end(), cmpx);
     xindex[0]=pos=0;
     for(i=1;i<SUBDIVISIONS;i++) {
         while((pos<pointcloud.pointcloud.size())&&(pointcloud.pointcloud[pos].x<pointcloud.pxmin+i*breitex)) {
             pos++;
         }
         xindex[i]=pos;
         if(pos!=xindex[i-1]) { //region is not empty, sort by y
             std::vector< cv::Point3d >::iterator start = pointcloud.pointcloud.begin();
             std::advance(start,xindex[i-1]);
             std::vector< cv::Point3d >::iterator end = pointcloud.pointcloud.begin();
             std::advance(end,xindex[i]);
             
             sort(start, end, cmpy);
         }
     }
     xindex[SUBDIVISIONS]=pointcloud.pointcloud.size();
     if(xindex[SUBDIVISIONS-1]<pointcloud.pointcloud.size()) {
         std::vector< cv::Point3d >::iterator start = pointcloud.pointcloud.begin();
         std::advance(start,xindex[SUBDIVISIONS-1]);
         sort(start, pointcloud.pointcloud.end(), cmpy);
     }
     
     //divide x regions with respect to y
     for(i=0;i<SUBDIVISIONS;i++) {
         pos=xindex[i];
         for(j=0;j<SUBDIVISIONS;j++) {
             pointcloud.pktindex[i][j].start=pos;
             while((pos<xindex[i+1])&&(pointcloud.pointcloud[pos].y<pointcloud.pymin+(j+1)*breitey)) {
                 pos++;
             }
             pointcloud.pktindex[i][j].ende=pos-1;
         }
         pointcloud.pktindex[i][SUBDIVISIONS-1].ende=xindex[i+1]-1;
     }
     std::cout << "quadtree of point cloud generated" << std::endl;
 }



void readLAS(const char * imagepath, cloud_t& pointcloud) {
    std::ifstream ifs;
    
    //todo: remove initialization if multiple point clouds should be used
    pointcloud.offset_x=pointcloud.offset_y=0;
    
    ifs.open(imagepath, std::ios::in | std::ios::binary);
    
    liblas::ReaderFactory f;
    liblas::Reader reader = f.CreateWithStream(ifs);
    
     
    liblas::Header const& header = reader.GetHeader();
    std::cout << "Compressed: " << header.Compressed() << std::endl;
    std::cout << "Signature: " << header.GetFileSignature() << '\n';
    std::cout << "Points count: " << header.GetPointRecordsCount() << '\n';
    
    bool first = true;
    
    while (reader.ReadNextPoint()) {
        liblas::Point const& p = reader.GetPoint();
        
        liblas::Classification c = p.GetClassification();
        uint8_t value = c.GetClass();
        
        if(value==18) { continue;} //ignore noise
        
        cv::Point3d punkt;
        punkt.x=p.GetX()-pointcloud.offset_x;
        punkt.y=p.GetY()-pointcloud.offset_y;
        punkt.z=p.GetZ();
        
        if(first) {
            first = false;
            pointcloud.offset_x=punkt.x;
            punkt.x=0;
            pointcloud.pxmax = 0;
            pointcloud.pxmin = 0;
            pointcloud.offset_y=punkt.y;
            pointcloud.pymax = 0;
            pointcloud.pymin = 0;
            punkt.y = 0;
        } else {
            if(punkt.x>pointcloud.pxmax) pointcloud.pxmax=punkt.x;
            if(punkt.x<pointcloud.pxmin) pointcloud.pxmin=punkt.x;
            if(punkt.y>pointcloud.pymax) pointcloud.pymax=punkt.y;
            if(punkt.y<pointcloud.pymin) pointcloud.pymin=punkt.y;
        }
        
        pointcloud.pointcloud.push_back(punkt);
    }
    std::cout << std::endl << "Number of points in cloud: " << (int)pointcloud.pointcloud.size() << std::endl;
    
    //todo: call quadtree generation after all redinging all point clouds if multiple point clouds should be used
    quadtree(pointcloud);
}

/*readDouble generates a double float from an ascii representation of a number*/
double readDouble(const char* str, long* pos) {
    double zahl = 0.0;
    double faktor=1.0;
    
    //ignore white spaces
    while(((str[*pos]<'0')||(str[*pos]>'9'))&&(str[*pos]!='-')&&(str[*pos]!=0)) (*pos)++;
    

    if(str[*pos]=='-') {
        faktor=-1.0; (*pos)++;
    }

    while((str[*pos]>='0')&&(str[*pos]<='9')) {
        zahl=10.0 * zahl+(double)(str[*pos]-'0');
        (*pos)++;
    }
    //*******************************************************************************************
    //remove leading numbers if more than 7 digits are used
    //*******************************************************************************************
    if(zahl>=10000000) {
        zahl=(double)(((long)zahl)%1000000);
    }
    if(str[*pos]=='.') /*decimal point*/
    {  (*pos)++;
        double nenner=10.0;
        while((str[*pos]>='0')&&(str[*pos]<='9')) {
            if(nenner<10000000.0) {
                zahl=zahl+ (double)(str[*pos]-'0')/nenner;
                nenner=nenner*10.0;
            }
            (*pos)++;
        }
    }
    return(faktor*zahl);
}


void readXYZ(const char * imagepath, cloud_t& pointcloud) {
    long i;
    
    //todo: remove initialization if multiple point clouds should be used
    pointcloud.offset_x=pointcloud.offset_y=0;
    
    printf("Read XYZ file: %s\n",imagepath);
    
    std::ifstream ifs;
    ifs.open(imagepath, std::ifstream::in);
    if (ifs.is_open())
    {
        ifs.seekg(0, ifs.end);
        long length = ifs.tellg();
        ifs.seekg(0, ifs.beg);
        
        char* filebuffer =  (char*) malloc(length+2);

        ifs.read(filebuffer, length);
        filebuffer[length]=0;
        
        i=0;
        
        bool first = true;
        
        while(i<length) {
            
            cv::Point3d punkt;
            punkt.x=readDouble(filebuffer,&i)-pointcloud.offset_x;
            i++; //ignore comma
            punkt.y=readDouble(filebuffer,&i)-pointcloud.offset_y;
            
            i++; //ignore comma
            punkt.z=readDouble(filebuffer,&i);
            
            //read until line break
            while((filebuffer[i]!=0)&&(filebuffer[i]!='\n')) i++;
            if(filebuffer[i]=='\n') i++;
            
            if(first) {
                first = false;
                pointcloud.offset_x=punkt.x;
                punkt.x=0;
                pointcloud.pxmax = 0;
                pointcloud.pxmin = 0;
                pointcloud.offset_y=punkt.y;
                pointcloud.pymax = 0;
                pointcloud.pymin = 0;
                punkt.y = 0;
            } else {
                if(punkt.x>pointcloud.pxmax) pointcloud.pxmax=punkt.x;
                if(punkt.x<pointcloud.pxmin) pointcloud.pxmin=punkt.x;
                if(punkt.y>pointcloud.pymax) pointcloud.pymax=punkt.y;
                if(punkt.y<pointcloud.pymin) pointcloud.pymin=punkt.y;
            }
            
            pointcloud.pointcloud.push_back(punkt);
        }
    
        ifs.close();
        free(filebuffer);
    }
    std::cout << std::endl << "Number of points in cloud: " << (int)pointcloud.pointcloud.size() << std::endl;
    //todo: call quadtree generation after all redinging all point clouds if multiple point clouds should be used
    quadtree(pointcloud);
}
