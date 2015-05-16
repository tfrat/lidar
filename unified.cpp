
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <string>
#include <pcl/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/grabcut_segmentation.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/surface/gp3.h>


using namespace pcl;
using namespace std;
//STRUCTS
typedef struct {
	pcl::ModelCoefficients::Ptr coeffs;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
} PlaneDetectOutput;

/*
 * LOOK HERE: A majority of these algorithms were adapted from tutorials found on the pcl documentation webpage.
 * Found here: http://pointclouds.org/documentation/tutorials/
 */

//Preprocessing
pcl::PointCloud<pcl::PointXYZ>::Ptr normalizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
pcl::PointXYZ mean_point (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
double normalize_cloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


// Filtering methods
pcl::PointCloud<pcl::PointXYZ>::Ptr filterPassthrough(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
pcl::PointCloud<pcl::PointXYZ>::Ptr filterRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
pcl::PointCloud<pcl::PointXYZ>::Ptr filterStat(pcl::PointCloud<pcl::PointXYZ>::Ptr input);

// Ground plane detection methods
PlaneDetectOutput planeEst(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int method, bool doSegment);


// Segmentation methods
pcl::PointCloud<pcl::PointXYZ>::Ptr useProgMorph(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
pcl::PointCloud<pcl::PointXYZ>::Ptr useMinCuts(pcl::PointCloud<pcl::PointXYZ>::Ptr fullCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr inliers);
pcl::PointCloud<pcl::PointXYZ>::Ptr useDiffOfNorm(pcl::PointCloud<pcl::PointXYZ>::Ptr input);
pcl::PointCloud<pcl::PointXYZ>::Ptr useCondEuclid(pcl::PointCloud<pcl::PointXYZ>::Ptr fullCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr inliers);
// Mesh generation methods
PolygonMesh generateMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);


int main (int argc, char** argv)
{

	// *********** Command line input *********** 
	if(argc!=8){
		std::cerr << "ENTER ALL 8 ARGUMENTS\n";
		std::cerr << "Usage: ./unified\n";
		std::cerr << "<input file>\n";
		std::cerr << "<output file>\n";
		std::cerr << "# of iterations\n";
		std::cerr << "<filter type> Radius, Stat, Pass, None\n";
		std::cerr << "<plane ext type>\n";
		std::cerr << "<segmentation type>  ProgMorph MinCuts\n";
		std::cerr << "<mesh type>  Not developed\n";
		return -1;
	}

	char *input = argv[1];
	char *output = argv[2];
	int numIter = atoi(argv[3]);
	char *filterType = argv[4];
	int planeExtType = atoi(argv[5]);
	char *segType = argv[6];
	char *meshType = argv[7];

	// timing variables
	clock_t t_part,t_total;
	float initTime,saveTime, filterTime,planeTime,segTime,meshTime;
	t_total = clock();
	t_part = clock();
	
	// *********** point cloud initialization ***********  
	// input point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	// filtered point cloud 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered;
	// plane inlier point clod 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane;
	// segmented ground point cloud 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented;
	
	// point cloud reader to read cloud from file
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ> (input, *cloud);

	initTime = (clock() - t_part)/CLOCKS_PER_SEC;
	std::cerr << "Cloud before filtering: " << std::endl; // display information on input cloud
	std::cerr << *cloud << std::endl;

	// normalize cloud 
	double range_max = normalize_cloud(cloud);

	// *********** Filter Point Cloud *********** 
	t_part = clock();
	
	// run filtering for the desired number of iterations, averaged for timing 
	for(int i = 0;i<numIter;++i){
		if(strcmp ("Radius",filterType)==0) {
			std::cout<< "Radius filter applied" <<endl;
			cloud_filtered = filterRadius(cloud);
		}else if(strcmp ("Stat",filterType)==0) {
			std::cout<< "Statistical filter applied" <<endl;
			cloud_filtered = filterStat(cloud);
		}else if(strcmp ("Pass",filterType)==0) {
			std::cout<< "Passthrough filter applied" <<endl;
			cloud_filtered = filterPassthrough(cloud);
		}else{
			std::cout<< "No filter applied" <<endl;
			cloud_filtered=cloud;
		}
	}

	filterTime = ((clock() - t_part)/(float)numIter)/CLOCKS_PER_SEC;
	std::cout<< "Filtering took (per iteration): " << filterTime <<endl;

	// *********** Plane Estimation  *********** 
	t_part = clock();
	if(planeExtType!= -1){ // if plane estimation is to be run
		PlaneDetectOutput pdOut;
		// run plane estimation for the desired number of iterations, averaged for timing 
		for(int i = 0;i<numIter;++i){ 
			pdOut = planeEst(cloud,planeExtType,true);
			cloud_plane = pdOut.cloud;
		}
	}
	planeTime = ((clock() - t_part)/(float)numIter)/CLOCKS_PER_SEC;

	std::cout<< "Plane Estimation took (per iteration): " << planeTime <<endl;

	// *********** Ground Plane Segmentation *********** 
	t_part = clock();
	// run Segmentation for the desired number of iterations, averaged for timing 
	for(int i = 0;i<numIter;++i){
		if(strcmp ("ProgMorph",segType)==0)
			cloud_segmented = useProgMorph(cloud_filtered);
		else if(strcmp ("MinCuts",segType)==0)
			cloud_segmented = useMinCuts(cloud_filtered,cloud_plane);
		else{
			cloud_segmented = cloud_filtered; 
		};
	}
	segTime = ((clock() - t_part)/(float)numIter)/CLOCKS_PER_SEC;
	std::cout<< "Ground segmentation took (per iteration): " <<segTime <<endl;
	
	
	std::stringstream outfile;
	outfile << argv[2] << "-" << filterType << "-" << planeExtType << "-" << segType;

	// *********** Mesh Generation *********** 
	if(strcmp("None", meshType) != 0) {	
		PolygonMesh triangles;

		t_part = clock();
		// run mesh genration for the desired number of iterations, averaged for timing 
		for(int i = 0;i<numIter;++i){
			triangles = generateMesh(cloud_segmented);
		}
		meshTime = ((clock() - t_part)/(float)numIter)/CLOCKS_PER_SEC;
		std::cout<< "Mesh Generation took (per iteration): " << meshTime <<endl;
		
		// output the mesh to a VTK file for visualization using the VTK viewer shoftware
		stringstream vtk_file;
		vtk_file << outfile.str() << ".vtk";
		pcl::io::saveVTKFile (vtk_file.str().c_str(), triangles);
	}
	
	// *********** Save Output *********** 
	
	// Save the segmented point cloud 
	pcl::PCDWriter writer;
	std::stringstream ss;
	ss << outfile.str() << ".pcd";
	writer.write<pcl::PointXYZ> (ss.str(), *cloud_segmented, false);
	saveTime = (clock() - t_part)/CLOCKS_PER_SEC;

	// Save the processing times to a csv file
	std::stringstream time_filename;
	time_filename << outfile.str() << ".csv";
	std::ofstream time_file;
	time_file.open(time_filename.str().c_str());
	time_file << "Filter, Plane Estimation, Segmentation, Mesh, Total" << std::endl;
	time_file << filterTime << ", " << planeTime  << ", " << segTime << ", " << meshTime << 
		", " << filterTime + planeTime + segTime + meshTime << std::endl;
	time_file.close();
	return (0);
}

/*
* filterPassthrough
*
* DESCRIPTION: Filter the input point cloud using the pass through (threshold) filtering 
*					method
* INPUTS: 
*			input - The input point cloud 
* RETURN VALUES
*			The filtered point cloud
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr filterPassthrough(pcl::PointCloud<pcl::PointXYZ>::Ptr input) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	// Create the filtering object
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud (input);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (-0.001, 0.001);
	pass.filter (*cloud_filtered);

	return cloud_filtered;
}

/*
* filterRadius
*
* DESCRIPTION: Filter the input point cloud using the radius filtering method
* INPUTS: 
*			input - The input point cloud 
* RETURN VALUES
*			The filtered point cloud
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr filterRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr input) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	// build the filter
	outrem.setInputCloud(input);
	outrem.setRadiusSearch(0.005);
	outrem.setMinNeighborsInRadius (2);
	// apply filter
	outrem.filter (*cloud_filtered);

	return cloud_filtered;
}

/*
* filterStat
*
* DESCRIPTION: Filter the input point cloud using the statistical filtering method
* INPUTS: 
*			input - The input point cloud 
* RETURN VALUES
*			The filtered point cloud
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr filterStat(pcl::PointCloud<pcl::PointXYZ>::Ptr input){

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud (input);
	sor.setMeanK (50);
	sor.setStddevMulThresh (1.0);
	sor.filter (*cloud_filtered);

	return cloud_filtered;
}

/*
* planeEst
*
* DESCRIPTION: Estimate the plane that best fits the data using various fitting methods.
* INPUTS: 
*			input - The input point cloud 
*			method - The method to use for fitting the plane
*			doSegment - A variable that determines if the point cloud should be segmentent
*						based on the points laying within a specified theshold from the 
*						calculated plane of best fit.  
* RETURN VALUES
*			The segmented point cloud, if desired
*/
PlaneDetectOutput planeEst(pcl::PointCloud<pcl::PointXYZ>::Ptr input, int method, bool doSegment){

	PlaneDetectOutput retval;

	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::SACSegmentation<pcl::PointXYZ> seg;

	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);

	switch (method){
		case 0:
			seg.setMethodType (pcl::SAC_RANSAC);
			break;
		case 1:
			seg.setMethodType (pcl::SAC_LMEDS);
			break;
		case 2:
			seg.setMethodType (pcl::SAC_MSAC);
			break;
		case 3:
			seg.setMethodType (pcl::SAC_RRANSAC);
			break;
		case 4:
			seg.setMethodType (pcl::SAC_RMSAC);
			break;
		case 5:
			seg.setMethodType (pcl::SAC_MLESAC);
			break;
		case 6:
			seg.setMethodType (pcl::SAC_PROSAC);
			break;
	}


	seg.setDistanceThreshold (0.001);

	seg.setInputCloud (input);
	seg.segment (*inliers, *coefficients);

	retval.coeffs = coefficients;
	if (inliers->indices.size () == 0)
	{
		PCL_ERROR ("Could not estimate a plane for dataset.");
		return retval;
	}
	if(doSegment){
		// Create the filtering object
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (input);
		extract.setIndices (inliers);
		extract.filter (*cloud_filtered);
		retval.cloud = cloud_filtered;
	}                          
	return retval;             
}       

/*
* useProgMorph
*
* DESCRIPTION: Segment the ground plane using the Progressive Morphological filtering 
*					method
* INPUTS: 
*			input - The input point cloud 
* RETURN VALUES
*			The segmented ground point cloud
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr useProgMorph(pcl::PointCloud<pcl::PointXYZ>::Ptr input){
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PointIndicesPtr ground (new pcl::PointIndices);
	// Create the filtering object
	pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
	pmf.setInputCloud (input);
	pmf.setMaxWindowSize (20);
	pmf.setSlope (1.0f);
	pmf.setInitialDistance (0.5f);
	pmf.setMaxDistance (3.0f);
	pmf.extract (ground->indices);

	// Create the filtering object
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud (input);
	extract.setIndices (ground);
	extract.filter (*cloud_filtered);

	return cloud_filtered;
}

/*
* useMinCuts
*
* DESCRIPTION: Segment the ground plane using the Min Cuts/ Max Flow (Graph cuts) 
*					segmentation method
* INPUTS: 
*			input - The input point cloud 
*			inliers - pointer to a point cloud to hold the inliers
* RETURN VALUES
*			The segmented ground point cloud
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr useMinCuts(pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointCloud<pcl::PointXYZ>::Ptr inliers){
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::MinCutSegmentation<pcl::PointXYZ> seg;
	pcl::IndicesPtr indices (new std::vector <int>);
	std::vector <pcl::PointIndices> clusters;
	seg.setInputCloud (input);

	seg.setForegroundPoints (inliers);

	seg.setSigma (0.1);
	seg.setRadius (0.05);
	seg.setNumberOfNeighbours (14);
	seg.setSourceWeight (0.5);

	seg.extract (clusters);;

	std::vector<pcl::PointIndices>::const_iterator it = (++(clusters.begin ())); 
	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
	{
		cloud_filtered->points.push_back (input->points[*pit]);
	}
	cloud_filtered->width = int (cloud_filtered->points.size ());
	cloud_filtered->height = 1;
	cloud_filtered->is_dense = true;

	return cloud_filtered;
}

/*
* generateMesh
*
* DESCRIPTION: Generate a triangulated mesh of the ground plane 
* INPUTS: 
*			cloud - The input point cloud 
* RETURN VALUES
*			The PolygonMesh containing the triangulation of the points
*/
PolygonMesh generateMesh(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
	int searchRad = 1;
	int mu = 5;
	pcl::PCLPointCloud2 cloud_blob;
	//* the data should be available in cloud

	// Estimate the normals of the points 
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud);
	n.setInputCloud (cloud);
	n.setSearchMethod (tree);
	n.setKSearch (20);
	n.compute (*normals);

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);

	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud_with_normals);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius (searchRad);

	// Set typical values for the parameters
	gp3.setMu (mu);
	gp3.setMaximumNearestNeighbors (200);
	gp3.setMaximumSurfaceAngle(M_PI/2);
	gp3.setMinimumAngle(M_PI/18);
	gp3.setMaximumAngle(2*M_PI/3);
	gp3.setNormalConsistency(false);


	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	return triangles;
}


/*
* mean_point
*
* DESCRIPTION: Calculate the mean value of a given point cloud
* INPUTS: 
*			cloud - The input point cloud 
* RETURN VALUES
*			The mean location (x,y,z) of all of the points in the point cloud
*/
pcl::PointXYZ mean_point (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	unsigned int i;
	unsigned int size = cloud->points.size();
	double mean_x=0.0, mean_y=0.0, mean_z=0.0;
	pcl::PointXYZ point;

	// sum up the location of all of the points
	for (i=0; i<size; i++)
	{
		mean_x += cloud->points[i].x;
		mean_y += cloud->points[i].y;
		mean_z += cloud->points[i].z;
	}
	
	// divide the summation in each dimension by the number of points (average)
	mean_x /= (double)size;
	mean_y /= (double)size;
	mean_z /= (double)size;

	//set the location of the mean point
	point.x = mean_x;
	point.y = mean_y;
	point.z = mean_z;

	return point;
}

/*
* normalize_cloud
*
* DESCRIPTION: Normalize a given point cloud to be contained in a unit cube
* INPUTS: 
*			cloud - The pointer to the input point cloud, point cloud altered in place 
* RETURN VALUES
*			The scaling factor that the cloud was scaled by.
*/
double normalize_cloud (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	unsigned int i;
	unsigned int size = cloud->points.size();
	pcl::PointXYZ mean_p;
	double min_x=0.0, min_y=0.0, min_z=0.0;
	double max_x=0.0, max_y=0.0, max_z=0.0;
	double range_x, range_y, range_z;
	double range_max = 0.0;

	// use above method to calculate the center of the point cloud
	mean_p = mean_point(cloud);

	// initialize the min and max to be the first point, insures min and max exist
	min_x = cloud->points[0].x-mean_p.x;
	min_y = cloud->points[0].y-mean_p.y;
	min_z = cloud->points[0].z-mean_p.z;
	max_x = min_x;
	max_y = min_y;
	max_z = min_x;
	
	// find the max and min points in each dimension (x, y, z)of the point cloud
	for (i=0; i<size; i++)
	{
		// center the point cloud on the origin
		cloud->points[i].x -=mean_p.x;
		cloud->points[i].y -=mean_p.y;
		cloud->points[i].z -=mean_p.z;

		// check if the current point is a minimum in any dimension 
		min_x = (cloud->points[i].x<min_x)?cloud->points[i].x:min_x;
		min_y = (cloud->points[i].y<min_y)?cloud->points[i].y:min_y;
		min_z = (cloud->points[i].z<min_z)?cloud->points[i].z:min_z;

		// check if the current point is a maximum in any dimension 
		max_x = (cloud->points[i].x>max_x)?cloud->points[i].x:max_x;
		max_y = (cloud->points[i].y>max_y)?cloud->points[i].y:max_y;
		max_z = (cloud->points[i].z>max_z)?cloud->points[i].z:max_z;
	}

	// calculate the range in each dimension
	range_x = abs(max_x -min_x);
	range_y = abs(max_y -min_y);
	range_z = abs(max_z -min_z);

	// find the dimension with the largest range
	range_max = (range_x>range_y)?range_x:range_y;
	range_max = (range_z>range_max)?range_z:range_max;

	
	//cout << 1.0 / range_max << " RANGE MAX " << endl;
	
	// normalize the point cloud
	for (i=0; i<size; i++)
	{
		cloud->points[i].x /= range_max;
		cloud->points[i].y /= range_max;
		cloud->points[i].z /= range_max;
	}

	return range_max;
}
