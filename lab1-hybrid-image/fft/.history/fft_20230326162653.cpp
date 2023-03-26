#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

cv::Mat ideal_Low_Pass_Filter(Mat &src, float sigma);
Mat ideal_lbrf_kernel(Mat &scr,float sigma);
Mat freqfilt(Mat &scr,Mat &blur);

int main( int argc, char *argv[])
{
    const char* filename = argc >=2 ? argv[1] : "/home/bill/Pictures/Lenna.png";

    Mat input = imread(filename, IMREAD_GRAYSCALE);
    if( input.empty())
        return -1;
    imshow("input",input);//显示原图

    Mat ideal = ideal_Low_Pass_Filter(input, 160);
    ideal = ideal(Rect(0,0, input.cols, input.rows));
    imshow("理想", ideal);
    cout << "input size: " << input.size() << " input type: " << input.type() << " input channels: " << input.channels() << endl;
    // Mat substracted = input - ideal;
    // imshow("理想减去", substracted);
    waitKey();
    return 0;
}

//*****************理想低通滤波器***********************
Mat ideal_lbrf_kernel(Mat &scr,float sigma)
{
    Mat ideal_low_pass(scr.size(),CV_32FC1); //，CV_32FC1
    float d0=sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
    for(int i=0;i<scr.rows ; i++ ){
        for(int j=0; j<scr.cols ; j++ ){
            double d = sqrt(pow((i - scr.rows/2),2) + pow((j - scr.cols/2),2));//分子,计算pow必须为float型
            if (d <= d0){
                ideal_low_pass.at<float>(i,j)=1;
            }
            else{
              ideal_low_pass.at<float>(i,j)=0;
            }
        }
    }
    string name = "理想低通滤波器d0=" + std::to_string(sigma);
    imshow(name, ideal_low_pass);
    return ideal_low_pass;
}

cv::Mat ideal_Low_Pass_Filter(Mat &src, float sigma)
{
    int M = getOptimalDFTSize(src.rows);
    int N = getOptimalDFTSize(src.cols);
    Mat padded;                 //调整图像加速傅里叶变换
    copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT, Scalar::all(0));
    padded.convertTo(padded,CV_32FC1); //将图像转换为flaot型

    Mat ideal_kernel=ideal_lbrf_kernel(padded,sigma);//理想低通滤波器
    Mat result = freqfilt(padded,ideal_kernel);
    return result;
}
//*****************频率域滤波*******************
Mat freqfilt(Mat &scr,Mat &blur)
{
    //***********************DFT*******************
    Mat plane[]={scr, Mat::zeros(scr.size() , CV_32FC1)}; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
    Mat complexIm;
    merge(plane,2,complexIm);//合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
    dft(complexIm,complexIm);//进行傅立叶变换，结果保存在自身

    //***************中心化********************
    split(complexIm,plane);//分离通道（数组分离）
//    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));//这里为什么&上-2具体查看opencv文档
//    //其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
    int cx=plane[0].cols/2;int cy=plane[0].rows/2;//以下的操作是移动图像  (零频移到中心)
    Mat part1_r(plane[0],Rect(0,0,cx,cy));  //元素坐标表示为(cx,cy)
    Mat part2_r(plane[0],Rect(cx,0,cx,cy));
    Mat part3_r(plane[0],Rect(0,cy,cx,cy));
    Mat part4_r(plane[0],Rect(cx,cy,cx,cy));

    Mat temp;
    part1_r.copyTo(temp);  //左上与右下交换位置(实部)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);  //右上与左下交换位置(实部)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    Mat part1_i(plane[1],Rect(0,0,cx,cy));  //元素坐标(cx,cy)
    Mat part2_i(plane[1],Rect(cx,0,cx,cy));
    Mat part3_i(plane[1],Rect(0,cy,cx,cy));
    Mat part4_i(plane[1],Rect(cx,cy,cx,cy));

     part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);

    //*****************滤波器函数与DFT结果的乘积****************
    Mat blur_r,blur_i,BLUR;
    multiply(plane[0], blur, blur_r); //滤波（实部与滤波器模板对应元素相乘）
    multiply(plane[1], blur,blur_i);//滤波（虚部与滤波器模板对应元素相乘）
    Mat plane1[]={blur_r, blur_i};
    merge(plane1,2,BLUR);//实部与虚部合并

      //*********************得到原图频谱图***********************************
    magnitude(plane[0],plane[1],plane[0]);//获取幅度图像，0通道为实部通道，1为虚部，因为二维傅立叶变换结果是复数
    plane[0]+=Scalar::all(1);  //傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
    log(plane[0],plane[0]);    // float型的灰度空间为[0，1])
    normalize(plane[0],plane[0],1,0,NORM_MINMAX);  //归一化便于显示
   imshow("原图像频谱图",plane[0]);

    idft( BLUR, BLUR);    //idft结果也为复数
    split(BLUR,plane);//分离通道，主要获取通道
    magnitude(plane[0],plane[1],plane[0]);  //求幅值(模)
    normalize(plane[0],plane[0],1,0,NORM_MINMAX);  //归一化便于显示
    return plane[0];//返回参数
}