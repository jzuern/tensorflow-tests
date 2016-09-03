// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.




 template <typename Dtype>

class BilinearFiller : public Filler<Dtype> {
public:
   explicit BilinearFiller(const FillerParameter& param)
       : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
     CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
     CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
     Dtype* data = blob->mutable_cpu_data();
     int f = ceil(blob->width() / 2.);
     float c = (2 * f - 1 - f % 2) / (2. * f);
     for (int i = 0; i < blob->count(); ++i) {
       float x = i % blob->width();
       float y = (i / blob->width()) % blob->height();
       data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
     }
     CHECK_EQ(this->filler_param_.sparse(), -1)
          << "Sparsity not supported by this Filler.";
   }
};