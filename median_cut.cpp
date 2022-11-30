//This module is used to find the dominant color of a texture

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

Retrieved from: http://en.literateprograms.org/Median_cut_algorithm_(C_Plus_Plus)?oldid=19175
 
iPattern: Point replaced by OpenCV cv::Vec3b, adjusted to find a dominant color
*/

#include <limits>
#include <queue>
#include <algorithm>
#include <list>
#include "median_cut.hpp"

class Block
{
    cv::Vec3b minCorner, maxCorner;
    cv::Vec3b* points;
    long pointsLength;
public:
    Block(cv::Vec3b* points, long pointsLength);
    cv::Vec3b * getPoints();
    long numPoints() const;
    int longestSideIndex() const;
    int longestSideLength() const;
    bool operator<(const Block& rhs) const;
    void shrink();
private:
    template <typename T>
    static T min(const T a, const T b)
    {
        if (a < b)
            return a;
        else
            return b;
    }

    template <typename T>
    static T max(const T a, const T b)
    {
        if (a > b)
            return a;
        else
            return b;
    }
};
template <int index>
class CoordinatePointComparator
{
public:
    bool operator()(cv::Vec3b left, cv::Vec3b right)
    {
        return left[index] < right[index];
    }
};


Block::Block(cv::Vec3b* points, long pointsLength)
{   //GBB data are copied unsorted:
    this->points = points;
    this->pointsLength = pointsLength;
    for(int i=0; i < 3; i++)
    {
        minCorner[i] = std::numeric_limits<unsigned char>::min();
        maxCorner[i] = std::numeric_limits<unsigned char>::max();
    }
}
cv::Vec3b * Block::getPoints()
{
    return points;
}

long Block::numPoints() const
{
    return pointsLength;
}
int Block::longestSideIndex() const
{
    int m = maxCorner[0] - minCorner[0];
    int maxIndex = 0;
    for(int i=1; i < 3; i++)
    {
        int diff = maxCorner[i] - minCorner[i];
        if (diff > m)
        {
            m = diff;
            maxIndex = i;
        }
    }
    return maxIndex;
}
int Block::longestSideLength() const
{
    int i = longestSideIndex();
    return maxCorner[i] - minCorner[i];
}
bool Block::operator<(const Block& rhs) const
{
    return this->longestSideLength() < rhs.longestSideLength();
}
void Block::shrink()
{
    int i,j;
    for(j=0; j<3; j++)
    {
        minCorner[j] = maxCorner[j] = points[0][j];
    }
    for(i=1; i < pointsLength; i++)
    {
        for(j=0; j<3; j++)
        {
            minCorner[j] = min(minCorner[j], points[i][j]);
            maxCorner[j] = max(maxCorner[j], points[i][j]);
        }
    }
}
std::list<cv::Vec3b> medianCut(cv::Vec3b* image, int numPoints, unsigned int desiredSize)
{
    std::priority_queue<Block> blockQueue;
    Block initialBlock(image, numPoints);
    initialBlock.shrink();
    blockQueue.push(initialBlock);
    while (blockQueue.size() < desiredSize)
    {
        Block longestBlock = blockQueue.top();
        blockQueue.pop();
        cv::Vec3b * begin  = longestBlock.getPoints();
        cv::Vec3b * median = longestBlock.getPoints() + (longestBlock.numPoints()+1)/2;
        cv::Vec3b * end    = longestBlock.getPoints() + longestBlock.numPoints();
        switch(longestBlock.longestSideIndex())
        {
            //partial sorting of the component with longest side:
            //nth replaces the element at position median by the element that would occur there
            //after sorting. Then this is used as a pivot element to sort other elements before
            //or after this position
            case 0: std::nth_element(begin, median, end, CoordinatePointComparator<0>()); break;
            case 1: std::nth_element(begin, median, end, CoordinatePointComparator<1>()); break;
            case 2: std::nth_element(begin, median, end, CoordinatePointComparator<2>()); break;
        }

        Block block1(begin, median-begin), block2(median, end-median);
        block1.shrink();
        block2.shrink();
        blockQueue.push(block1);
        blockQueue.push(block2);
    }
    std::list<cv::Vec3b> result;
    while(!blockQueue.empty())
    {
        Block block = blockQueue.top();
        blockQueue.pop();
        cv::Vec3b * points = block.getPoints();

        int sum[3] = {0};
        for(int i=0; i < block.numPoints(); i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum[j] += points[i][j];
            }
        }
        
        //GBB
        if(block.numPoints()>0) {

            cv::Vec3b averagePoint;
            for(int j=0; j < 3; j++)
            {
                averagePoint[j] = sum[j] / block.numPoints();
            }

            result.push_back(averagePoint);
        }
    }
    return result;
}

bool mean_color(cv::Mat &texture, cv::Mat &mask, float &red, float &green, float &blue) {
    int anz=8; //the number of colors is reduced to "anz". Then for each of these colors, the pixels are counted.
    //The dominant color is the reduced color with the largest number of pixels
    
    //cv::imshow("Texturbild",texture);
    //cv::imshow("Maske",mask);
    
    std::vector< cv::Vec3b > points;

    for (int j=0; j<texture.rows;j++)
        for (int i=0; i<texture.cols; i++)
        {
            if(mask.at< unsigned char >(j,i)) {
                cv::Vec3b temp=texture.at<cv::Vec3b>(j,i);
                cv::Vec3b p(temp[0], temp[1], temp[2]);
                points.push_back(p);
            }
        }
    
    if(points.size()==0) return false;
    
    if((int)points.size()<anz) anz=(int)points.size();

    std::list<cv::Vec3b> palette =
    medianCut(&points[0], (int)points.size(), anz);

    std::list<cv::Vec3b>::iterator iter;
    
    int occurence[palette.size()];
    cv::Vec3b occuranceColor[palette.size()];
    for(int j=0; j<palette.size(); j++) occurence[j]=0;
    int jj=0;
    for (iter = palette.begin() ; iter != palette.end(); iter++)
    {
        occuranceColor[jj] = *iter;
        occurence[jj]=0;
        jj++;
    }
    
    //Determine occurence of colors
    
    for (int j=0; j<texture.rows; j++)
    {
        for (int i=0; i<texture.cols; i++)
        {
            if(mask.at< unsigned char >(j,i)) {
                int min=500000;
                cv::Vec3b color;
                int pos=0, minpos=0;
                for (iter = palette.begin() ; iter != palette.end(); iter++)
                {
                    cv::Vec3b temp=texture.at<cv::Vec3b>(j,i);
                    cv::Vec3b palette_color = *iter;
                    float abst=sqrt(((int)palette_color[0]-temp[0])*((int)palette_color[0]-temp[0])+
                        ((int)palette_color[1]-temp[1])*((int)palette_color[1]-temp[1])+
                        ((int)palette_color[2]-temp[2])*((int)palette_color[2]-temp[2]));
                    if (abst < min)
                    {
                        min=abst;
                        color[0]=(int)palette_color[0];
                        color[1]=(int)palette_color[1];
                        color[2]=(int)palette_color[2];
                        minpos=pos;
                    }
                    pos++;
                }
                //texture.at<cv::Vec3b>(j,i) = cv::Vec3b(color[0], color[1],color[2]);
                occurence[minpos]++;
            }
            
        }
    }
    //cv::imshow("Median Cut",texture);
    //cv::waitKey(0);
    
    //-----------------------------------------------------------------------------------------
    
    int maxpos=0; int maximum=0;
    for(int j=0; j<palette.size(); j++) {
        if(occurence[j] > maximum) {
            maximum = occurence[j];
            maxpos = j;
        }
    }

    blue  = ((double)occuranceColor[maxpos][0])/255.0;
    green = ((double)occuranceColor[maxpos][1])/255.0;
    red   = ((double)occuranceColor[maxpos][2])/255.0;
       
    return true;
}


