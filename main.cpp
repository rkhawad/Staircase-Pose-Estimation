#include <cv.h>
#include <lsd.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ios>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;

struct xy_pair{
	int x1;
	int y1;
	int x2;
	int y2;
};

bool acompare(xy_pair lhs, xy_pair rhs){
	return lhs.y1 < rhs.y1;
}
bool bcompare(xy_pair lhs, xy_pair rhs){
	return lhs.y2 < rhs.y2;
}

Point2f lineintersect2(Vec4i line1, Vec4i line2)
{
	Point2f pt;
	if(line1[0]!= line1[2] && line2[0]!=line2[2])
	{
		float m1,m2,c,d;
		m1 = (float)(line1[3] - line1[1]) / (line1[2] - line1[0]);
		c = (float)(line1[1] - (m1 * line1[0]));
		m2 = (float)(line2[3] - line2[1]) / (line2[2] - line2[0]);
		d = (float)line2[1] - (m2 * line2[0]);
		pt.x = (d-c)/(m1-m2);
		cout << "slope1 " << m1 << " slope2 " << m2 << "\n";
		cout << "intercept1 " << c << " intercept2 " << d << "\n";
	//	pt.y = c;
	//	pt.x = (pt.y - d)/m2;
	        pt.y = (m1*pt.x) + c;
	}
	else
	{
		pt.x = line2[0];
		pt.y = line1[1];
	}
	return pt;
}

Point2f lineintersect(Vec4i line1,Vec4i line2)
{
	Point2f pt;
	double K_Nr_1,K_Nr_2,K_Dr,K;
	int x1,y1,x2,y2,x3,y3,x4,y4;

	x1=line1[0];
	y1=line1[1];
	x2=line1[2];
	y2=line1[3];
	x3=line2[0];
	y3=line2[1];
	x4=line2[2];
	y4=line2[3];
	double midx=(x1+x2)/2;
	double midy=(y1+y2)/2;
	K_Nr_1 = (y4-y3)*(midx-x3);
	K_Nr_2 = (x4-x3)*(midy-y3);
	K_Dr = pow((y4-y3),2)+pow((x4-x3),2);
	K = (K_Nr_1 - K_Nr_2)/K_Dr;
	pt.x = midx - (K*(y4-y3)) ;
	pt.y = midy + (K*(x4-x3)) ;
	//cout << pt << endl;
	return pt;
}
Vec4i drawStraightLine(cv::Mat *img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color,cv::Point2f p3,cv::Point2f p4)
{
	Vec4i l;
	Point2f p, q;
	// Check if the line is a vertical line because vertical lines don't have slope
	if (p1.x != p2.x)
	{
	//	q.x = p4.x;
		q.y = p4.y;
		p.x = p3.x;
		// Slope equation (y1 - y2) / (x1 - x2)
		float m = (p1.y - p2.y) / (p1.x - p2.x);
		// Line equation:  y = mx + b
		float b = p1.y - (m * p1.x);
		q.x = (q.y - b)/m;

		p.y = p3.y;
		p.x = (p.y - b)/m;
	/*	if(p.x > img->cols)
		{
			cout << "yes" << "\n";
			p.x = img->cols-1;
		}*/
	//	cout << p1.x << " " << p1.y << " " << p2.x << " " << p2.y << " " << p3.x << " " << p3.y << "\n";
	//	cout << p.x << " " << p.y << " " << q.x << " " << q.y << "\n";

	}
	else
	{
		p.x = q.x = p2.x;
		//	p.y = 0;
		p.y = p3.y;
		//	q.y = img->rows;
		q.y = p4.y;
	}
	l[0] = p.x;
	l[1] = p.y;
	l[2] = q.x;
	l[3] = q.y;
	cv::line(*img, p, q, color, 3);
	return l;
}
Vec4i drawhStraightLine(cv::Mat *img, cv::Point2f p1, cv::Point2f p2, cv::Scalar color,cv::Point2f p3,cv::Point2f p4)
{
	Vec4i l;
	Point2f p, q;
	// Check if the line is a vertical line because vertical lines don't have slope
	if (p1.x != p2.x)
	{
		//p.x = 0;
		p.x = p3.x ;
		if(p.x < 0)
			p.x = 0;
		//q.x = img->cols;
		q.x = p4.x;
		if(q.x >= img->cols)
			q.x = img->cols - 1;
		// Slope equation (y1 - y2) / (x1 - x2)
		float m = (p1.y - p2.y) / (p1.x - p2.x);
		// Line equation:  y = mx + b
		//m = m*(-1);
		float b = p1.y - (m * p1.x);
		p.y = m*p.x + b;
		q.y = m*q.x + b;

	}
	else
	{
		p.x = q.x = p2.x;
		//      p.y = 0;
		p.y = p3.y;
		//      q.y = img->rows;
		q.y = p4.y;
	}
	l[0] = p.x;
	l[1] = p.y;
	l[2] = q.x;
	l[3] = q.y;
	cv::line(*img, p, q, color, 2);
	return l;
}

int main(int argc , char **argv)
{
	if(argc!=4)
	{
		cout << "Usage:lsd imagename" << endl;
		return 0;
	}
	int binsize = atoi(argv[2]);
	Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat src2,src3,src4,src5,src6,src_gray,tmp;
	src.copyTo(src2);
	src.copyTo(src3);
	src.copyTo(src4);
	src.copyTo(src5);
	src.copyTo(src6);
	cvtColor(src,tmp,CV_RGB2GRAY);
	tmp.convertTo(src_gray, CV_64FC1);
	int cols = src_gray.cols;
	int rows = src_gray.rows;
	vector<Vec4i> lines;
	vector<Vec4i> lines_hor;
	vector<Vec4i> lines_hor_sel;
	xy_pair num[1000];
	xy_pair numv[2000];
	xy_pair structure[1000];

	image_double image = new_image_double(cols, rows);
	image->data = src_gray.ptr<double>(0);
	ntuple_list ntl = lsd(image);

	Mat lsd = Mat::zeros(rows, cols, CV_8UC1);
	Point pt1,pt2;
	int cnt =0;int tempx =0;int tempy = 0;int midv = 0;
	int cntv = 0;

	for(int j=0;j!=ntl->size;j++)
	{
		pt1.x = ntl->values[0 + j*ntl->dim];   // All the detected lines
		pt1.y = ntl->values[1 + j*ntl->dim];
		pt2.x = ntl->values[2 + j*ntl->dim];
		pt2.y = ntl->values[3 + j*ntl->dim];
		double width = ntl->values[4 + j*ntl->dim];

		line(src, pt1, pt2, Scalar(0,255,0),1,CV_AA);
		float num = pt2.y - pt1.y;
		float den  = pt2.x - pt1.x;
		double theta= (atan2(num,den)*180/CV_PI);
		// Horizontal Lines
		if((abs(theta) >=0 && abs(theta)<=10) || (abs(theta)>=170 && abs(theta)<=180 ))
		{
			Vec4i l;
			l[0] = pt1.x;
			l[1] = pt1.y;
			l[2] = pt2.x;
			l[3] = pt2.y;

			lines.push_back(l);
			cnt++;

			line(src3, pt1, pt2, Scalar(0,255,0), 1, CV_AA);
		}
		// Verical Lines
		else if((abs(theta) >= 85 && abs(theta) <= 95) || (abs(theta) >= -95 && abs(theta) <= -85))
		{
			if(sqrt(pow((pt2.x-pt1.x),2)+pow((pt2.y-pt1.y),2)) > 30)
			{
				midv = (pt1.y + pt2.y)/2;
				if(pt2.y < midv)
				{
					tempx = pt1.x;     tempy = pt1.y;
					pt1.x = pt2.x;     pt1.y = pt2.y;
					pt2.x = tempx;     pt2.y = tempy;
				}
				numv[cntv].x1 = pt1.x;
				numv[cntv].y1 = pt1.y;
				numv[cntv].x2 = pt2.x;
				numv[cntv].y2 = pt2.y;
			//	cout << numv[cntv].x1 << " " << numv[cntv].y1 << " " << numv[cntv].x2 << " " << numv[cntv].y2 << "\n";
				line(src5,pt1,pt2,Scalar(121,255,0),1,CV_AA);
				cntv++;
			}
		}
	}
	std::sort(numv, numv+cntv, bcompare);
	double lines_len[1000];
	double max_len = 0;
	for(int j=0;j<cnt;j++)
	{
		Vec4i l = lines[j];
		lines_len[j] = sqrt(pow((l[2]-l[0]),2) + pow((l[3]-l[1]),2));
		if(lines_len[j]>max_len)
			max_len=lines_len[j];
	}
	int cnt_hor = 0;
	double temp_len = 0.3 * max_len;
	for(int j=0;j<cnt;j++)
	{
		if(lines_len[j] > temp_len)
		{
			Vec4i l;
			l = lines[j];
			lines_hor.push_back(l);
			cnt_hor++;
		}
	}
	int hist_acc[1000];
	for(int j=0;j<cnt_hor;j++)
	{
		Vec4i l;
		l = lines_hor[j];
	}
	for(int i=0;i<cnt_hor;i++)
		for(int k=0;k<cnt_hor;k++)
		{
			if(i!=k)
			{
				Point2f pt = lineintersect(lines_hor[i],lines_hor[k]);
				Vec4i l = lines_hor[k];
				if(round(pt.x) > min(l[0],l[2]) && round(pt.x) < max(l[0],l[2]) )
				{
					//	line( I, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,0), 1, CV_AA);
					int x=round(pt.x);
					int y=round(pt.y);
					hist_acc[i]++;
				}
			}
		}
	double cnt_sel=0;
	int n =0;
	int mid1 =0;
	int numf[100][100];
	for(int k=0;k<cnt_hor;k++)
	{
		if(hist_acc[k]>0.30*cnt_hor)
		{
			Vec4i l=lines_hor[k];
			lines_hor_sel.push_back(lines_hor[k]);
			cnt_sel++;
		}
	}
	if(cnt_sel > 3)
	{
		for(int k=0;k<cnt_sel;k++)
		{
			Vec4i l;
			l = lines_hor_sel[k];
			num[n].x1 = l[0];
			num[n].y1 = l[1];
			num[n].x2 = l[2];
			num[n].y2 = l[3];
			mid1 = (l[0] + l[2])/2;
			if(num[n].x2 < mid1)
			{
				tempx = num[n].x1;     tempy = num[n].y1;
				num[n].x1 = num[n].x2; num[n].y1 = num[n].y2;
				num[n].x2 = tempx;     num[n].y2 = tempy;
			}
			//cout << num[n].x1 << " " << num[n].y1 << " " << num[n].x2 << " " << num[n].y2 << "\n";
			n++;
			line(src4,Point(l[0],l[1]),Point(l[2],l[3]),Scalar(255,0,255),2,CV_AA);
		}
	}
	std::sort(num,num+n,acompare);
//	cout << n << "\n";
	double m,k;
	xy_pair numfinal[100];
	xy_pair numfinal1[100];
	m = num[0].y1;
	k = num[n-1].y1;
	cout << m << " " << k << "\n";
	int freq[binsize+1];
	for(int i=0;i<binsize;i++)
		freq[i] = 0;
	double bin;
	bin = (k - m)/binsize;
	cout << bin << "\n";
	double set =0;
	int j = 0;
	for(int i = 0;i<n;i++)
	{
		set = m;
		for(j=0;j<binsize;j++)
		{
			if((num[i].y1 >= set+(j*bin)) && (num[i].y1 < set+((j+1)*bin)))
			{
				numf[j][freq[j]] = i;
				freq[j]++;
			}
		}
	}
	int z = 0;
	int in, out;
	for(int i=0;i<binsize;i++)
	{
		if(freq[i]!=0)
		{
			in = num[numf[i][0]].x1;
			out = num[numf[i][0]].x2; 
			for(j=0;j<freq[i];j++)
			{
				if(num[numf[i][j]].x1 <= in)
				{
					numfinal[z].x1 = num[numf[i][j]].x1;
					numfinal[z].y1 = num[numf[i][j]].y1;
				}
				if(num[numf[i][j]].x2 >= out)
				{
					numfinal[z].x2 = num[numf[i][j]].x2;
					numfinal[z].y2 = num[numf[i][j]].y2;
				}
			}
			z++;
		}
	}
	rows = src.rows;
	cols = src.cols;
	std::sort(numfinal,numfinal+z,acompare);
	cout << "total lines " << z << "\n";
	Point p1,p2;
	Point m1,m2;
	Point mm1,mm2;
	Point n1,n2;
	Mat subMat;
	int max_dist2 = 0;
	//	Point mid1,mid2;
	double dist;
	double max_dist =0;
	Vec4i extended_lines;
	Vec4i extended_lines2;
	Vec4i l1;
	Vec4i l2;
	int include[1000];
	for(int i=0;i<z-1;i++)
	{
		 int middlex1 = (numfinal[i].x1 + numfinal[i].x2)/2;
		 int middlex2 = (numfinal[i+1].x1 + numfinal[i+1].x2)/2;
		 int middley1 = (numfinal[i].y1 + numfinal[i].y2)/2;
		 int middley2 = (numfinal[i+1].y1 + numfinal[i+1].y2)/2;
		 dist = sqrt(pow((middlex2-middlex1),2)+pow((middley2-middley1),2));
		 cout << i << " dist " << dist << "\n"; 
		 if(dist < 100)
		 {
			 include[i] = 0;
			 include[i+1] = 0;
		 }
		 else
		 {
			 include[i] = 1;
			 include[i+1] = 1;
		 } 
	}
	int cnth = 0;
	for(int i =0;i<z;i++)
	{
		if(include[i]!=1)
		{
			numfinal1[cnth].x1 = numfinal[i].x1;
			numfinal1[cnth].y1 = numfinal[i].y1;
			numfinal1[cnth].x2 = numfinal[i].x2;
		        numfinal1[cnth].y2 = numfinal[i].y2;
			
			line( src2, p1, p2, Scalar(255,0,255), 3, CV_AA);
			stringstream s;
			s << dist;
			string stry = s.str();
			putText(src2, stry,p1,FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(200,200,250), 1, CV_AA);
			cnth++;
		}
	}
	for(int i=0;i<cnth;i++)
	{
		dist = numfinal1[i].x2 - numfinal1[i].x1;
		p1.x = numfinal1[i].x1;
		p1.y = numfinal1[i].y1;
		p2.x = numfinal1[i].x2;
		p2.y = numfinal1[i].y2;
		if(dist > max_dist){
			max_dist2 = max_dist;
			max_dist = dist;
			m1.x = p1.x;
			m1.y = p1.y;
			m2.x = p2.x;
			m2.y = p2.y;
		}
		else if(dist > max_dist2)
		{
			max_dist2 = dist;
			mm1.x = p1.x;
			mm1.y = p1.y;
			mm2.x = p2.x;
			mm2.y = p2.y;
		}
	}
//	circle(src6, Point(numfinal1[cnth-1].x1,rows-1),10, Scalar(255,2,255),CV_FILLED, 8,0); 
//	circle(src6, Point(numfinal1[cnth-1].x2,rows-1),10, Scalar(255,2,255),CV_FILLED, 8,0);
	int x = cols-1;
	int y = rows-1;
	circle(src6, Point(x,y),10, Scalar(255,0,0),CV_FILLED, 8,0);
	cout << rows << " " << cols << "\n";
	 
//	l1 =  drawStraightLine(&src2,m1,mm1,Scalar(255,123,123),Point(numfinal1[cnth-1].x1,rows-1),Point(numfinal1[0].x1,numfinal1[0].y1));

	l1 =  drawStraightLine(&src2,m1,Point(numfinal1[0].x1,numfinal1[0].y1),Scalar(255,123,123),Point(numfinal1[cnth-1].x1,rows-1),Point(numfinal1[0].x1,numfinal1[0].y1));
	circle(src6, Point(l1[0],l1[1]),10, Scalar(0,255,2),CV_FILLED, 8,0);
	cout << "point1 " << l1[0] << " " << l1[1] << "\n";
	cout << "point2 " << l1[2] << " " << l1[3] << "\n";
//	l2 = drawStraightLine(&src2, m2,mm2,Scalar(255,123,123),Point(numfinal1[cnth-1].x2,rows-1),Point(numfinal1[0].x2,numfinal1[0].y2)); 
	
	l2 = drawStraightLine(&src2, m2,Point(numfinal1[0].x2,numfinal1[0].y2), Scalar(255,123,123),Point(numfinal1[cnth-1].x2,rows-1),Point(numfinal1[0].x2,numfinal1[0].y2));
 	cout << "point3 " << l2[0] << " " << l2[1] << "\n";
        cout << "point4 " << l2[2] << " " << l2[3] << "\n";
	circle(src6, Point(l1[2],l1[3]),10, Scalar(0,255,2),CV_FILLED, 8,0);
	circle(src6, Point(l2[0],l2[1]),10, Scalar(0,255,2),CV_FILLED, 8,0);
	circle(src6, Point(l2[2],l2[3]),10, Scalar(0,255,2),CV_FILLED, 8,0);
	//	circle(src6, Point(l1[2],l2[3]),10, Scalar(255,2,2),CV_FILLED, 8,0);
//	circle(src6, Point(l2[0],l2[1]),10, Scalar(255,2,255),CV_FILLED, 8,0);
//	circle(src6, Point(l2[2],l2[3]),10, Scalar(2,255,2),CV_FILLED, 8,0);
	int potcnt = 0;
	for(int i =0;i<cnth;i++)
	{
		p1.x = numfinal1[i].x1;
		p1.y = numfinal1[i].y1;
		p2.x = numfinal1[i].x2;
		p2.y = numfinal1[i].y2;
		
	//	circle(src6, Point(l1[0],l1[1]),10, Scalar(255,237,25),CV_FILLED, 8,0);
	//	circle(src6, Point(l1[2],l1[3]),10, Scalar(255,237,25),CV_FILLED, 8,0);
	//	circle(src6, Point(l2[0],l2[1]),10, Scalar(255,237,25),CV_FILLED, 8,0);
	//      circle(src6, Point(l2[2],l2[3]),10, Scalar(255,237,25),CV_FILLED, 8,0);
	
		Vec4i l;
		l[0] = l1[0];
		l[1] = l1[1];
		l[2] = l1[2];
		l[3] = l1[3];

		extended_lines = drawhStraightLine(&src2, p1, p2, Scalar(255,0,255),Point(0,m1.y),Point(cols,m2.y));
		cout << "Extended " << extended_lines[0] << " " << extended_lines[1] << " " << extended_lines[2] << " " << extended_lines[3] <<"\n";
		
		Point2f intersection = lineintersect2(extended_lines,l1); 
		circle(src2, intersection,5, Scalar(255,0,25),CV_FILLED, 8,0);
		cout << "PoI1 " << intersection.x << " " << intersection.y << "\n";

		l[0] = l2[0];
		l[1] = l2[1];
		l[2] = l2[2];
		l[3] = l2[3];

		Point2f intersection2 = lineintersect2(extended_lines,l2);
		circle(src2, intersection2,5, Scalar(255,0,25),CV_FILLED, 8,0);
		line( src6, intersection, intersection2, Scalar(255,0,255), 3, CV_AA);
		cout << "PoI2 " << intersection2.x << " " << intersection2.y << "\n";
		structure[potcnt].x1 = intersection.x;
		structure[potcnt].y1 = intersection.y;
		structure[potcnt].x2 = intersection2.x;
		structure[potcnt].y2 = intersection2.y;
		potcnt++;
	}
	cout << "nopoint " << potcnt << "\n";
	line(src,m1,m2,Scalar(255,123,123),3,CV_AA);
//	line(src2,m1,Point(numfinal1[0].x1,numfinal1[0].y1),Scalar(255,123,123),3,CV_AA);
	
	//drawStraightLine(&src2,m1,Point(numfinal[0].x1,numfinal[0].y1),Scalar(255,123,123),Point(numfinal[z-1].x1,rows),Point(numfinal[0].x1,numfinal[0].y1));
	
//	line(src2,m2,Point(numfinal1[0].x2,numfinal1[0].y2),Scalar(255,123,123),3,CV_AA);
	cout << rows << "\n";
	
	//drawStraightLine(&src2, m2,Point(numfinal[0].x2,numfinal[0].y2), Scalar(255,123,123),Point(numfinal[z-1].x2,rows),Point(numfinal[0].x2,numfinal[0].y2));
	

	for(int i=0;i<cnth-1;i++)
	{
		line( src2, Point(numfinal1[i].x1,numfinal1[i].y1), Point(numfinal1[i+1].x1,numfinal1[i+1].y1), Scalar(0,0,255), 2, CV_AA);
		line( src2, Point(numfinal1[i].x2,numfinal1[i].y2), Point(numfinal1[i+1].x2,numfinal1[i+1].y2), Scalar(0,0,255), 2, CV_AA);
	}
	for(int i=0;i<potcnt-1;i++)
	{
	        line( src6, Point(structure[i].x1,structure[i].y1), Point(structure[i+1].x1,structure[i+1].y1), Scalar(0,0,255), 2, CV_AA);
	        line( src6, Point(structure[i].x2,structure[i].y2), Point(structure[i+1].x2,structure[i+1].y2), Scalar(0,0,255), 2, CV_AA);
	}
	float middlex1 = (structure[0].x1 + structure[0].x2)/2;
	float middley1 = (structure[0].y1 + structure[0].y2)/2;
	float middlex2 = (structure[potcnt-1].x1 + structure[potcnt-1].x2)/2;
	float middley2 = (structure[potcnt-1].y1 + structure[potcnt-1].y2)/2;
	line( src6, Point(middlex1,middley1), Point(middlex2,middley2), Scalar(0,0,255), 2, CV_AA);
	float numer = middley2 - middley1;
	float denom  = middlex2 - middlex1;
	double theta= (atan2(numer,denom)*180/CV_PI);
	stringstream t;
	t << theta;
	string stry = t.str();
	putText(src6, stry,Point(middlex1,middley1),FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(200,200,250), 2, CV_AA);
	
	int maximum = 0;
	int count1 = 0;
	int vx,vy = 0;
	for(int i=0;i<z;i++)
	{
		int x = (numfinal1[i].x1 + numfinal1[i].x2)/2;
		int y = (numfinal1[i].y1 + numfinal1[i].y2)/2;
		int check = max_dist;
		for(j=0;j<cntv;j++)
		{
			vx = (numv[j].x1 + numv[j].x2)/2;
			vy = (numv[j].y1 + numv[j].y2)/2;
			dist = sqrt(pow((x-vx),2)+pow((y-vy),2));
			if(dist < check)
			{
				check = dist;
				n1.x = numv[j].x1;
				n1.y = numv[j].y1;
				n2.x = numv[j].x2;
				n2.y = numv[j].y2;
			}
		}
		if(sqrt(pow((n1.x-n2.x),2)+pow((n1.y-n2.y),2)) > maximum )
		         maximum = sqrt(pow((n1.x-n2.x),2)+pow((n1.y-n2.y),2));
		count1++;
		stringstream s;
		s << count1;
		string stry = s.str();
		putText(src2, stry, n1,FONT_HERSHEY_COMPLEX_SMALL,1, cvScalar(200,200,250), 1, CV_AA);
		line( src2, n1, n2, Scalar(0,0,0),3, CV_AA);
	}
	Point m3,m4;
	m3.x = m1.x;
	m3.y = m1.y + maximum;
	m4.x = m2.x;
	m4.y = m2.y + maximum;
	line( src2, m1, m2, Scalar(0,0,0), 3, CV_AA);
	line( src2, m3, m4, Scalar(0,0,0), 3, CV_AA);
	line( src2, m1, m3, Scalar(0,0,0), 3, CV_AA);
	line( src2, m2, m4, Scalar(0,0,0), 3, CV_AA);


	Size size(500,500);//the dst image size,e.g.100x100
	Mat dst;//dst image
	resize(src6,dst,size);//resize image

	cv::namedWindow("src", CV_WINDOW_AUTOSIZE);
	cv::imshow("src", src);
	cv::namedWindow("src2", CV_WINDOW_AUTOSIZE);
	cv::imshow("src2", src2);
	imwrite("second.jpg",src2);
	cv::namedWindow("src3", CV_WINDOW_AUTOSIZE);
	cv::imshow("src3",src3);
	imwrite("third.jpg",src5);
	cv::namedWindow("src4", CV_WINDOW_AUTOSIZE);
	cv::imshow("src4",src4);
	cv::namedWindow("src5",CV_WINDOW_AUTOSIZE);
	cv::imshow("src5",src5);
	cv::namedWindow("src6",CV_WINDOW_AUTOSIZE);
	cv::imshow("src6",dst);

	cv::waitKey(0);
	stringstream ssresult;
	ssresult << argv[3];
	string s1 =  ssresult.str() + ".jpg";
	imwrite(s1,src2);
	String st = "_str";
	
	ssresult << st;
	string s12 =  ssresult.str() + ".jpg";
	imwrite(s12,src6);

	return 0;
}
