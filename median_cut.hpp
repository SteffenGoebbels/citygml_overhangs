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

*/

#ifndef _MEDIAN_CUT_H_
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

bool mean_color(cv::Mat &texture, cv::Mat &mask, float &red, float &green, float &blue);

#endif /* #ifndef _MEDIAN_CUT_H_ */
