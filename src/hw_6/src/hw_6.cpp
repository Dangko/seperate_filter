#include <stdlib.h>
#include "ros/ros.h"
#include <iostream>
#include<time.h>
#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

void Mean_filter(Mat input,Mat output,int n);
void Mean_filter_separate(Mat input,Mat output,int n);
void Mean_filter_separate_X(Mat input,Mat output,int n);
void Mean_filter_separate_Y(Mat input,Mat output,int n);

int main(int argc,char** argv)
{
    VideoCapture capture;
    capture.open(0);
    if(!capture.isOpened())
    {
        printf("capture isn't opened!");
        return 0;
    }
    Mat frame;
    Mat src = imread("/home/dango/dango_file/machine visual/h6_ws/src/hw_6/test_img/cat.jpeg");
    Mat src_gray;
    cvtColor(src,src_gray,CV_BGR2GRAY);

    Mat output_3 = src_gray.clone();
    Mat output_3_sep = src_gray.clone();
    Mat output_5 = src_gray.clone();
    Mat output_5_sep = src_gray.clone();

    Mat output_7_X = src_gray.clone();
    Mat output_7_Y = src_gray.clone();
    Mat output_7 = src_gray.clone();

    clock_t start_time1=clock();
    Mean_filter(src_gray,output_3,3);
    clock_t end_time1=clock();

    clock_t start_time2=clock();
    Mean_filter_separate(src_gray,output_3_sep,3);
    clock_t end_time2=clock();

    clock_t start_time3=clock();
    Mean_filter(src_gray,output_5,5);
    clock_t end_time3=clock();

    clock_t start_time4=clock();
    Mean_filter_separate(src_gray,output_5_sep,5);
    clock_t end_time4=clock();

    Mean_filter_separate_X(src_gray,output_7_X,7);
    Mean_filter_separate_Y(src_gray,output_7_Y,7);
    Mean_filter_separate(src_gray,output_7,7);
    cout<<"The time of mean_filter with 3x3 kernel is "<<(end_time1-start_time1)*1.0/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    cout<<"The time of mean_filter_separate with 3x3 kernel is "<<(end_time2-start_time2)*1.0/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    cout<<"The time of mean_filter with 5x5 kernel is "<<(end_time3-start_time3)*1.0/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    cout<<"The time of mean_filter_separate with 5x5 kernel is "<<(end_time4-start_time4)*1.0/CLOCKS_PER_SEC*1000<<"ms"<<endl;

    imshow("original",src_gray);
    imshow("3x3 mean_filter",output_3);
    imshow("3x3 mean_filter_separate",output_3_sep);
    imshow("5x5 mean_filter",output_5);
    imshow("5x5 mean_filter_separate",output_5_sep);
    imshow("7x7 mean_filter X",output_7_X);
    imshow("7x7 mean_filter Y",output_7_Y);
    imshow("7x7 mean_filter",output_7);

    waitKey(5);
    while(1)
    {
        waitKey(5);
    }

    return 0;
}


void Mean_filter(Mat input,Mat output,int n)
{
    double** tem;
    tem = new double*[n];
    for(int i=0;i<n;i++)
    {
        tem[i] = new double[n];
    }
    int xi = (n-1)/2;
    int yi = xi;
    for(int i= 0;i<n;i++)
    {
        for(int j= 0;j<n;j++)
        {
            tem[i][j] = 1.0/(n*n);
        }
    }

    int m = (n-1)/2;
    int rows = input.rows;
    int cols = input.cols;
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            double sum = 0;
            for(int xj = i-m;xj<=i+m;xj++)
            {
                for(int yj =j-m;yj<=j+m;yj++)
                {
                    if(xj<0||xj>=rows||yj<0||yj>=cols)
                    {
                        continue;
                    }
                    sum += input.at<uchar>(xj,yj)*tem[xj-i+m][yj-j+m];
                }
            }
            output.at<uchar>(i,j) = uchar(sum);
        }
    }
}

void Mean_filter_separate(Mat input,Mat output,int n)
{
    double* kernel_n;
    kernel_n = new double[n];

    for(int i=0;i<n;i++)
    {
        kernel_n[i] = 1.0/n;
    }

    int m = (n-1)/2;
    int rows = input.rows;
    int cols = input.cols;

    Mat temp = output.clone();

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            double sum = 0;
            for(int xj = j-m;xj<=j+m;xj++)
            {
                if(xj<0||xj>=cols)
                {
                    continue;
                }
                sum += input.at<uchar>(i,xj)*kernel_n[xj-j+m];
            }
            temp.at<uchar>(i,j) = uchar(sum);
        }
    }

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            double sum = 0;
            for(int xi = i-m;xi<=i+m;xi++)
            {
                if(xi<0||xi>=rows)
                {
                    continue;
                }
                sum += temp.at<uchar>(xi,j)*kernel_n[xi-i+m];
            }
            output.at<uchar>(i,j) = uchar(sum);
        }
    }
}

void Mean_filter_separate_X(Mat input,Mat output,int n)
{
    double* kernel_n;
    kernel_n = new double[n];

    for(int i=0;i<n;i++)
    {
        kernel_n[i] = 1.0/n;
    }

    int m = (n-1)/2;
    int rows = input.rows;
    int cols = input.cols;


    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            double sum = 0;
            for(int xj = j-m;xj<=j+m;xj++)
            {
                if(xj<0||xj>=cols)
                {
                    continue;
                }
                sum += input.at<uchar>(i,xj)*kernel_n[xj-j+m];
            }
            output.at<uchar>(i,j) = uchar(sum);
        }
    }
}

void Mean_filter_separate_Y(Mat input,Mat output,int n)
{
    double* kernel_n;
    kernel_n = new double[n];

    for(int i=0;i<n;i++)
    {
        kernel_n[i] = 1.0/n;
    }

    int m = (n-1)/2;
    int rows = input.rows;
    int cols = input.cols;

    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            double sum = 0;
            for(int xi = i-m;xi<=i+m;xi++)
            {
                if(xi<0||xi>=rows)
                {
                    continue;
                }
                sum += input.at<uchar>(xi,j)*kernel_n[xi-i+m];
            }
            output.at<uchar>(i,j) = uchar(sum);
        }
    }
}

