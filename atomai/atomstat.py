import numpy as np
from sklearn import mixture
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

class imlocal:

    def __init__(self,
                 network_output, 
                 coord_all, 
                 r,
                 coord_class):
        self.network_output = network_output
        self.nb_classes = network_output.shape[-1]
        self.coord_all = coord_all
        self.coord_class = coord_class
        self.r = r
        (self.imgstack, 
         self.imgstack_com, 
         self.imgstack_frames) = self.extract_subimages()
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape
        self.X_vec = self.imgstack.reshape(self.d0, self.d1*self.d2*self.d3)

       
    def extract_subimages(self):
        """
        Extracts subimages centered at certain atom class/type
        in the neural network output

        Args:
            imgdata: 4D numpy array
                Prediction of a neural network with dimensions 
                (batch_size x height x width x channels)
            coord: N x 3 numpy array
                (x, y, class) coordinates data  
            d: int
                defines size of a square subimage

        Returns:
            stack of subimages, (x, y) coordinates of their centers    
        """
        imgstack, imgstack_com, imgstack_frames = [], [], []
        for i, (img, coord) in enumerate(
            zip(self.network_output, self.coord_all.values())):
            c = coord[np.where(coord[:,2]==self.coord_class)][:,0:2]
            img_cr_all, com = self._extract_subimages(img, c, self.r)
            imgstack.append(img_cr_all)
            imgstack_com.append(com)
            imgstack_frames.append(np.ones(len(com), int) * i)
        imgstack = np.concatenate(imgstack, axis=0)
        imgstack_com = np.concatenate(imgstack_com, axis=0)
        imgstack_frames = np.concatenate(imgstack_frames, axis=0)
        return imgstack, imgstack_com, imgstack_frames

    @classmethod
    def _extract_subimages(cls,
                           imgdata,
                           coord,
                           r):
        """
        Extracts subimages centered at specified coordinates
        for a single image

        Args:
            imgdata: 3D numpy array
                Prediction of a neural network with dimensions 
                (height x width x channels)
            coord: N x 2 numpy array
                (x, y) coordinates  
            r: int
                square image side = 2*r

        Returns:
            stack of subimages, (x, y) coordinates of their centers    
        """
        img_cr_all = []
        com = []
        for c in coord:
            cx = int(np.around(c[0]))
            cy = int(np.around(c[1]))
            img_cr = np.copy(
                imgdata[cx-r:cx+r, cy-r:cy+r, :])
            if img_cr.shape[0:2] == (int(r*2), int(r*2)):
                img_cr_all.append(img_cr[None, ...])
                #com.append(np.array([cx, cy])[None, ...])
                com.append(c[None, ...])
                
        img_cr_all = np.concatenate(img_cr_all, axis=0)
        com = np.concatenate(com, axis=0)
        return img_cr_all, com

    def gmm(self, 
            n_components, 
            covariance='diag', 
            random_state=1,
            plot_results=False):
        """
        Applies Gaussian mixture model to image stack

        Args:
            n_components: int
                number of components
            covariance: str
                type of covariance ('full', 'diag', 'tied', 'spherical')
            random_state: int
                random state instance
            plot_results: bool
                plotting gmm components
            
        Returns:
            cla: 3D numpy array
                First dimension correspond to individual mixture component
            classes: 1D numpy array
                labels for every subimage in image stack
        """
        clf = mixture.GaussianMixture(
            n_components=n_components, 
            covariance_type=covariance, 
            random_state=random_state)
        classes = clf.fit_predict(self.X_vec) + 1
        cla = np.ndarray(shape=(
            np.amax(classes), int(self.r*2), int(self.r*2), self.nb_classes))
        if plot_results:
            rows = int(np.ceil(float(n_components)/5))
            cols = int(np.ceil(float(np.amax(classes))/rows))
            fig = plt.figure(figsize=(4*cols, 4*(1+rows//2)))
            gs = gridspec.GridSpec(rows, cols)
            print('GMM components')
        for i in range(np.amax(classes)):
            cl = self.imgstack[classes == i + 1]
            cla[i] = np.mean(cl, axis=0)
            if plot_results:
                ax = fig.add_subplot(gs[i])
                if self.nb_classes == 3:
                    ax.imshow(cla[i], Interpolation='Gaussian')
                elif self.nb_classes == 1:
                    ax.imshow(cla[i, :, :, 0], Interpolation='Gaussian')
                else:
                    raise NotImplementedError(
                        "Can plot only images with 3 and 1 channles")
                ax.axis('off')
                ax.set_title('Class '+str(i+1)+'\nCount: '+str(len(cl)))       
        if plot_results:
            plt.subplots_adjust(hspace=0.6, wspace=0.4)
            plt.show()
        return cla, classes


    @classmethod
    def get_trajectory(cls, 
                       coord_class_dict, 
                       ck, 
                       rmax):
        flow = np.empty((0, 3))
        frames = []
        c0 = ck
        for k, c in coord_class_dict.items():
            d, index = spatial.cKDTree(
                c[:,:2]).query(c0, distance_upper_bound=rmax)
            if d != np.inf:
                flow = np.append(flow, [c[index]], axis=0)
                frames.append(k)
                c0 = c[index][:2]
        return flow, np.array(frames)

    def get_all_trajectories(self, 
                             n_components, 
                             covariance='diag', 
                             random_state=1,
                             rmax=10):
        classes = self.gmm(
            n_components, covariance, random_state)[1]
        coord_class_dict = {
            i: np.concatenate(
                (self.imgstack_com[np.where(self.imgstack_frames==i)[0]],
                    classes[np.where(self.imgstack_frames==i)[0]][..., None]),
                    axis=-1) 
            for i in range(int(np.ptp(self.imgstack_frames)+1))
        }
        all_trajectories = []
        all_frames = []
        for ck in coord_class_dict[0][:, :2]:
            flow, frames = self.get_trajectory(coord_class_dict, ck, rmax)
            all_trajectories.append(flow)
            all_frames.append(frames)
        return all_trajectories, all_frames

    @classmethod
    def renumerate_classes(cls, classes):
        diff = np.unique(classes) - np.arange(len(np.unique(classes)))
        diff_d = {cl: d for d, cl in zip(diff, np.unique(classes))}
        classes_renum = [cl - diff_d[cl] for cl in classes]
        return np.array(classes_renum, dtype=np.int64)

    def transition_matrix(self,
                          n_components, 
                          covariance='diag', 
                          random_state=1,
                          rmax=10):
        trajectories_all, frames_all = self.get_all_trajectories(
            n_components, covariance, random_state, rmax)
        transitions_all = []
        for traj in trajectories_all:
            classes = self.renumerate_classes(traj[:, -1])
            m = transitions(classes).calculate_transition_matrix()
            transitions_all.append(m)
        return transitions_all, trajectories_all, frames_all


class transitions:
    """
    Calculates and displays (optionally) Markov transition matrix
    Args:
        trace: 1D numpy array or list
            sequence of states/classes
    """
    def __init__(self, trace):
        self.trace = trace

    def calculate_transition_matrix(self, 
                                    plot_results=False, 
                                    plot_values=False):
        """
        Calculates Markov transition matrix
        Args:
            plot_results: bool
                plot calculated transition matrix
            plot_values: bool
                show calculated transition rates
        Returns: 
            m: 2D numpy array
                calculated transition matrix
        """
        n = 1 + max(self.trace) # number of states
        M = np.zeros(shape=(n, n))  
        for (i,j) in zip(self.trace, self.trace[1:]):
            M[i][j] += 1
        # now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        if plot_results:
            self.plot_transition_matrix(M, plot_values)
        return M

    @classmethod
    def plot_transition_matrix(cls,
                               m,
                               plot_values=False):
        """
        Plots transition matrix
        Args:
            m: 2D numpy array
                transition matrix
            plot_values: bool
                show claculated transtion rates
        """
        print('Transition matrix')
        m_ = np.concatenate((np.zeros((1, m.shape[1])), m), axis=0)
        m_ = np.concatenate((np.zeros((m_.shape[0], 1)), m_), axis=1)
        xt = np.arange(len(m) + 1)
        yt = np.arange(len(m) + 1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.imshow(m_, cmap='Reds')
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.set_xticklabels(xt, fontsize=10)  
        ax.set_yticklabels(xt, fontsize=10)  
        ax.set_xlabel('Transition class', fontsize=18)
        ax.set_ylabel('Starting class', fontsize=18)
        ax.set_xlim(0.5, len(m)+0.5)
        ax.set_ylim(len(m)+0.5, 0.5)
        if plot_values:
            for (j,i),label in np.ndenumerate(m_):
                if i !=0 and j !=0:
                    ax.text(i,j, "%.2f" % label,
                            ha='center', va='center', fontsize=6)
        plt.show()
