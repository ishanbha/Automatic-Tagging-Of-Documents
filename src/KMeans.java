import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;



public class KMeans extends Configured implements Tool  {
	
	/* this method returns initial clusters */
	public double[][] initialPoints(String fileName) throws Exception {
		Scanner fileIn = new Scanner(new FileReader(fileName));
		double[][] points = new double[10][58];
		int i=0,j;
		while(fileIn.hasNext()){
			String p[] = fileIn.nextLine().split(" ");
			for(j=0;j<p.length;j++){
				points[i][j] = Double.parseDouble(p[j]);
			}
			i++;
		}
		fileIn.close();
		
		return points;
	}
	
	
	/*	this method finds out the cosine-similarity between two points
	    the formula is cos theta = summation(a*b)/summation(a)summation(b)*/
	public double cosim(double[] p1, double[] p2){
		int i;
		double num=0.0,den1=0.0,den2=0.0;
		for(i=0;i<p1.length;i++){
			num += p1[i]*p2[i];
			den1 += p1[i]*p1[i];
			den2 += p2[i]*p2[i];
		}
		double sim = (num/Math.sqrt(den1))/Math.sqrt(den2);
		return sim;
		
	}
	
	
	@SuppressWarnings("deprecation")
	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		Configuration conf = new Configuration();
		conf.set("clust","clust.txt");
					
		/*	Iterates as many times as the number of passes of K-Means	*/
		for(int i=0;i<20;i++){
			Job job = new Job(conf);
			job.setJarByClass(KMeans.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
	
			job.setMapperClass(Map1.class);
			job.setReducerClass(Reduce1.class);
			
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
	
			FileInputFormat.addInputPath(job, new Path(args[0]));
			FileOutputFormat.setOutputPath(job, new Path(args[1]));
			
			job.waitForCompletion(true);
			
			FileUtils.deleteDirectory(new File("output"));
		}
		return 0;
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		int res = ToolRunner.run(new Configuration(), new KMeans(), args);
		
		System.exit(res);
		
	}
}


class Map1 extends Mapper<LongWritable, Text, IntWritable, Text> {
	KMeans km = new KMeans();
	boolean flag = true;
	double[][] clusters;
	double cost = 0.0;
	
	
	/*	Evaluates the cost of each iteration after each map phase 
	 and stores it in a file named cost	*/
	@Override
	protected void cleanup(Context context) throws IOException{
		FileWriter fw = new FileWriter("cost",true);
		fw.write(cost+"\n");
		fw.close();
	}
	
	@Override
	public void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		int i;
		String clustFile;
		/*	get the initial cluster points	*/
		if(flag){
			try {
				clustFile = context.getConfiguration().get("clust");
				clusters = km.initialPoints(clustFile);
			} catch (Exception e) {
			}
			flag = false;
		}
		
		/*	Points converted to double array	*/
		String p[] = value.toString().split(" ");
		double point[] = new double[58];
		for(i=0;i<p.length;i++)
			point[i] = Double.parseDouble(p[i]);
		double max = 0.0;
		int maxindex=0;
		
		/*	find the max similarity (minimum distance) cluster for the point	*/
		for(i=0;i<clusters.length;i++){
			double sim = km.cosim(point,clusters[i]);
			if(sim > max){
				max = sim;
				maxindex = i;
			}
		}
		
		/*	determine cost related with this iteration	*/
		for(i=0;i<point.length;i++){
			cost += Math.pow((clusters[maxindex][i] - point[i]),2);
		}
		
		/*	write each point as value and the cluster it is similar to
		 as key	*/
		context.write(new IntWritable (maxindex+1), value);
		
	}
}


class Reduce1 extends Reducer<IntWritable, Text, IntWritable, Text> {
	int index = 1;
	boolean flag = true;
	FileWriter pw;
	String clu="";
	
	/*	method to output the tags related to each
	 cluster in an increasing order based on their frequency value	*/
	@Override
	protected void cleanup(Context context) throws FileNotFoundException{
		double[][] clusters = new double[10][58];
		String lines[] = clu.split("\n");
		
		String vocab[] = new String[58];
		/*	load the vocab file containing individual words	*/
		Scanner in = new Scanner(new FileReader("vocab.txt"));
		for(int i=0;i<58;i++)
			vocab[i] = in.nextLine();
		in.close();
		
		/*	cluster points converted to double matrix	*/
		for(int i=0;i<lines.length;i++){
			String col[] = lines[i].split(" ");
			for(int j=0;j<col.length;j++){
				clusters[i][j] = Double.parseDouble(col[j]);
			}
		}
		
		/*	Storing the cluster points in a tree map to sort the indexes	*/
		for(int i=0;i<clusters.length;i++){
			TreeMap<Double,Integer> map = new TreeMap<Double,Integer>();
			for(int j=0;j<clusters[i].length;j++){
				map.put(clusters[i][j], j);
			}
			/*	displaying the tags for each cluster in increasing order of frequency	*/
			for(Map.Entry<Double, Integer> entry : map.entrySet()){
				int indx = entry.getValue();
				System.out.print(vocab[indx]+" ");
			}
			System.out.println();
		}
	}
	
	
	
	@Override
	public void reduce(IntWritable key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
		
		/*	delete the clust file to update it with new clusters	*/
		if(flag){
			String clust = context.getConfiguration().get("clust");
			File file = new File(clust);
			if(file.exists())
				file.delete();
			/*	true is given to ensure file is appended and not overwritten	*/
			pw = new FileWriter(clust, true);
			flag = false;
		}
		
		/*	update and append new cluster points	*/
		double[] newPoints = new double[58];
		String cluster="";
		int count=0;
		Iterator<Text> val = values.iterator();
		
		/*	update the cluster center using the average of all points in the cluster	*/
		while(val.hasNext()){
			String p[] = val.next().toString().split(" ");
			for(int j=0;j<p.length;j++){
				newPoints[j] += Double.parseDouble(p[j]);
			}
			count++;
		}
		int i;
		for(i=0;i<newPoints.length-1;i++){
			cluster += Double.toString(newPoints[i]/count) + " ";
		}
		
		cluster += Double.toString(newPoints[i]/count) +"\n";
		pw.write(cluster);
		index++;
		
		/*	update the variable 'clu' to be used in the cleanup method	*/
		clu += cluster;
		if(index == 11){
			pw.close();
		}
	}
}