import numpy as np


class Orbitsphere:
    def __init__(self):
        self.X_78 = []
        self.Y_78 = []
        self.Z_78 = []
        self.X_80 = []
        self.Y_80 = []
        self.Z_80 = []
        self.len_80 = []
        self.len_78 = []

    def Orbitsphere_78_exact(self, M, N, r, phi):
        # ----------------------------------------- Page 78 -----------------------------------------
        for m in range(1, M):
            M_term = (m * 2 * np.pi) / M
            convolution_matrix1 = [[(1 / 4) * (1 + 3 * np.cos(M_term)),
                                    (1 / 4) * (-1 + np.cos(M_term) + 2 * np.sqrt(2) * np.sin(M_term)),
                                    (1 / 4) * (-np.sqrt(2) + np.sqrt(2) * np.cos(M_term) - 2 * np.sin(M_term))],
                                   [(1 / 4) * (-1 + np.cos(M_term) - 2 * np.sqrt(2) * np.sin(M_term)),
                                    (1 / 4) * (1 + 3 * np.cos(M_term)),
                                    (1 / 4) * (np.sqrt(2) - np.sqrt(2) * np.cos(M_term) - 2 * np.sin(M_term))],
                                   [(1 / 2) * (((-1 + np.cos(M_term)) / np.sqrt(2)) + np.sin(M_term)),
                                    (1 / 4) * (np.sqrt(2) - np.sqrt(2) * np.cos(M_term) + 2 * np.sin(M_term)),
                                    (np.cos(M_term / 2)) ** 2]]
            for n in range(1, N):
                N_term = (n * 2 * np.pi) / N
                convolution_matrix2 = [[(1 / 2) + (np.cos(N_term) / 2),
                                        -(1 / 2) + (np.cos(N_term) / 2),
                                        -(np.sin(N_term) / np.sqrt(2))],
                                       [-(1 / 2) + (np.cos(N_term) / 2),
                                        (1 / 2) + (np.cos(N_term) / 2),
                                        -(np.sin(N_term) / np.sqrt(2))],
                                       [(np.sin(N_term) / np.sqrt(2)),
                                        (np.sin(N_term) / np.sqrt(2)),
                                        (np.cos(N_term))]]

                M1 = [np.linspace(0, 0, len(phi)), r*np.cos(phi), -r*np.sin(phi)]
                Y00 = np.dot(convolution_matrix1, (np.dot(convolution_matrix2, M1)))

                self.X_78.append(Y00[0])
                self.Y_78.append(Y00[1])
                self.Z_78.append(Y00[2])

        self.len_78 = len(self.X_78)

    def Orbitsphere_80_exact(self, M, N, r, phi):
        # ----------------------------------------- Page 80 -----------------------------------------
        for m in range(1, M):
            M_term = (m * 2 * np.pi) / M
            convolution_matrix3 = [[(1 / 4) * (1 + 3 * np.cos(M_term)),
                                    (1 / 4) * (-1 + np.cos(M_term) + 2 * np.sqrt(2) * np.sin(M_term)),
                                    (1 / 4) * (-np.sqrt(2) + np.sqrt(2) * np.cos(M_term) - 2 * np.sin(M_term))],
                                   [(1 / 4) * (-1 + np.cos(M_term) - 2 * np.sqrt(2) * np.sin(M_term)),
                                    (1 / 4) * (1 + 3 * np.cos(M_term)),
                                    (1 / 4) * (np.sqrt(2) - np.sqrt(2) * np.cos(M_term) - 2 * np.sin(M_term))],
                                   [(1 / 2) * (((-1 + np.cos(M_term)) / np.sqrt(2)) + np.sin(M_term)),
                                    (1 / 4) * (np.sqrt(2) - np.sqrt(2) * np.cos(M_term) + 2 * np.sin(M_term)),
                                    (np.cos(M_term / 2)) ** 2]]
            for n in range(1, N):
                N_term = (n * 2 * np.pi) / N
                convolution_matrix4 = [[(np.cos(N_term) / 2) - (np.sin(N_term) / 2),
                                        (np.sin(N_term) / np.sqrt(2)) + (np.cos(N_term) / np.sqrt(2)),
                                        (np.cos(N_term) / 2) - (np.sin(N_term) / 2)],
                                       [-(np.cos(N_term) / 2) - (np.sin(N_term) / 2),
                                        -(np.sin(N_term) / np.sqrt(2)) + (np.cos(N_term) / np.sqrt(2)),
                                        -(np.cos(N_term) / 2) - (np.sin(N_term) / 2)],
                                       [(-1 / np.sqrt(2)),
                                        0,
                                        (1 / np.sqrt(2))]]

                M2 = [np.linspace(0, 0, len(phi)), r*np.cos(phi), -r*np.sin(phi)]
                Y00 = np.dot(convolution_matrix3, (np.dot(convolution_matrix4, M2)))

                self.X_80.append(Y00[0])
                self.Y_80.append(Y00[1])
                self.Z_80.append(Y00[2])

        self.len_80 = len(self.X_80)
