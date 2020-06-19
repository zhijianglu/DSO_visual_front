//
// Created by lab on 20-3-25.
//

#ifndef DSO_MODULES_PIXELSELECTOR_H
#define DSO_MODULES_PIXELSELECTOR_H
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "PixelGradient.h"
#include "Utils.h"

using namespace cv;
using namespace std;
typedef PixelGradient FrameHessian;


class PixelSelector
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int makeMaps(
      const FrameHessian* const fh,
      float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

  PixelSelector(int w, int h);
  ~PixelSelector();
  int currentPotential; 		//!< 当前选择像素点的潜力, 就是网格大小, 越大选点越少


  bool allowFast;
  void makeHists(const FrameHessian* const fh);
 private:

  Eigen::Vector3i select(const FrameHessian* const fh,
                         float* map_out, int pot, float thFactor=1);


  unsigned char* randomPattern;


  int* gradHist;  			//!< 根号梯度平方和分布直方图, 0是所有像素个数
  float* ths;					//!< 平滑之前的阈值
  float* thsSmoothed;			//!< 平滑后的阈值
  int thsStep;
  const FrameHessian* gradHistFrame;
};


//########### TODO to select good point

template<int pot>
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac)
{

  memset(map_out, 0, sizeof(bool)*w*h);

  int numGood = 0;
  for(int y=1;y<h-pot;y+=pot)  // 每隔一个pot遍历
  {
    for(int x=1;x<w-pot;x+=pot)
    {
      int bestXXID = -1; // gradx 最大
      int bestYYID = -1; // grady 最大
      int bestXYID = -1; // gradx-grady 最大
      int bestYXID = -1; // gradx+grady 最大

      float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

      Eigen::Vector3f* grads0 = grads+x+y*w; // 当前网格的起点
      // 分别找到该网格内上面4个best
      for(int dx=0;dx<pot;dx++)
        for(int dy=0;dy<pot;dy++)
        {
          int idx = dx+dy*w;
          Eigen::Vector3f g=grads0[idx]; // 遍历网格内的每一个像素
          float sqgd = g.tail<2>().squaredNorm(); // 梯度平方和
          float TH = THFac*minUseGrad_pixsel * (0.75f);  //阈值, 为什么都乘0.75 ? downweight

          if(sqgd > TH*TH)
          {
            float agx = fabs((float)g[1]);
            if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

            float agy = fabs((float)g[2]);
            if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

            float gxpy = fabs((float)(g[1]-g[2]));
            if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

            float gxmy = fabs((float)(g[1]+g[2]));
            if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
          }
        }

      bool* map0 = map_out+x+y*w; // 选出来的像素为TRUE

      // 选上这些最大的像素
      if(bestXXID>=0)
      {
        if(!map0[bestXXID]) // 没有被选
          numGood++;
        map0[bestXXID] = true;

      }
      if(bestYYID>=0)
      {
        if(!map0[bestYYID])
          numGood++;
        map0[bestYYID] = true;

      }
      if(bestXYID>=0)
      {
        if(!map0[bestXYID])
          numGood++;
        map0[bestXYID] = true;

      }
      if(bestYXID>=0)
      {
        if(!map0[bestYXID])
          numGood++;
        map0[bestYXID] = true;

      }
    }
  }

  return numGood;
}

//* 同上, 只是把pot作为参数
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac)
{

  memset(map_out, 0, sizeof(bool)*w*h);

  int numGood = 0;
  for(int y=1;y<h-pot;y+=pot)
  {
    for(int x=1;x<w-pot;x+=pot)
    {
      int bestXXID = -1;
      int bestYYID = -1;
      int bestXYID = -1;
      int bestYXID = -1;

      float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

      Eigen::Vector3f* grads0 = grads+x+y*w;
      for(int dx=0;dx<pot;dx++)
        for(int dy=0;dy<pot;dy++)
        {
          int idx = dx+dy*w;
          Eigen::Vector3f g=grads0[idx];
          float sqgd = g.tail<2>().squaredNorm();
          float TH = THFac*minUseGrad_pixsel * (0.75f);

          if(sqgd > TH*TH)
          {
            float agx = fabs((float)g[1]);
            if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

            float agy = fabs((float)g[2]);
            if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

            float gxpy = fabs((float)(g[1]-g[2]));
            if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

            float gxmy = fabs((float)(g[1]+g[2]));
            if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
          }
        }

      bool* map0 = map_out+x+y*w;

      if(bestXXID>=0)
      {
        if(!map0[bestXXID])
          numGood++;
        map0[bestXXID] = true;

      }
      if(bestYYID>=0)
      {
        if(!map0[bestYYID])
          numGood++;
        map0[bestYYID] = true;

      }
      if(bestXYID>=0)
      {
        if(!map0[bestXYID])
          numGood++;
        map0[bestXYID] = true;

      }
      if(bestYXID>=0)
      {
        if(!map0[bestYXID])
          numGood++;
        map0[bestYXID] = true;

      }
    }
  }

  return numGood;
}


inline int makePixelStatus(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft=5, float THFac = 1)
{
  if(sparsityFactor < 1) sparsityFactor = 1; // 网格的大小, 在网格内选择最大的

  int numGoodPoints;


  if(sparsityFactor==1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
  else if(sparsityFactor==2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
  else if(sparsityFactor==3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
  else if(sparsityFactor==4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
  else if(sparsityFactor==5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
  else if(sparsityFactor==6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
  else if(sparsityFactor==7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
  else if(sparsityFactor==8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
  else if(sparsityFactor==9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
  else if(sparsityFactor==10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
  else if(sparsityFactor==11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
  else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


  /*
   * #points is approximately proportional to sparsityFactor^2.
   */

  float quotia = numGoodPoints / (float)(desiredDensity);

  int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f; // 更新网格大小


  if(newSparsity < 1) newSparsity=1;


  float oldTHFac = THFac;
  if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;  // 已经是最小的了, 但是数目还是不够, 就减小阈值

  // 如果满足网格大小变化小且阈值是0.5 || 点数量在20%误差内 || 递归次数已到 , 则返回
  if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
      ( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
      recsLeft == 0)
  {

//		printf(" \n");
    //all good
    sparsityFactor = newSparsity;
    return numGoodPoints;
  }
  else // 否则进行递归
  {
//		printf(" -> re-evaluate! \n");
    // re-evaluate.
    sparsityFactor = newSparsity;
    return makePixelStatus(grads, map, w,h, desiredDensity, recsLeft-1, THFac);
  }
}

//########### TODO 初始化帧结构

struct Pnt
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // index in jacobian. never changes (actually, there is no reason why).
  float u,v;

  // idepth / isgood / energy during optimization.
  float idepth;				//!< 该点对应参考帧的逆深度
  bool isGood;				//!< 点在新图像内, 相机前, 像素值有穷则好
  Vec2f_ energy;				//!< [0]残差的平方, [1]正则化项(逆深度减一的平方) // (UenergyPhotometric, energyRegularizer)
  bool isGood_new;
  float idepth_new;			//!< 该点在新的一帧(当前帧)上的逆深度
  Vec2f_ energy_new;			//!< 迭代计算的新的能量

  float iR;					//!< 逆深度的期望值
  float iRSumNum;				//!< 子点逆深度信息矩阵之和

  float lastHessian;			//!< 逆深度的Hessian, 即协方差, dd*dd
  float lastHessian_new;		//!< 新一次迭代的协方差

  // max stepsize for idepth (corresponding to max. movement in pixel-space).
  float maxstep;				//!< 逆深度增加的最大步长

  // idx (x+y*w) of closest point one pyramid level above.
  int parent;		  			//!< 上一层中该点的父节点 (距离最近的)的id
  float parentDist;			//!< 上一层中与父节点的距离

  // idx (x+y*w) of up to 10 nearest points in pixel space.
  int neighbours[10];			//!< 图像中离该点最近的10个点
  float neighboursDist[10];   //!< 最近10个点的距离

  float my_type; 				//!< 第0层提取是1, 2, 4, 对应d, 2d, 4d, 其它层是1
  float outlierTH; 			//!< 外点阈值
};


#endif //DSO_MODULES_PIXELSELECTOR_H
