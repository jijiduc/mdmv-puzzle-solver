class Config:
    """Configuration globale des paramètres avec capacités d'adaptation"""
    
    def __init__(self):
        self.DEBUG = False
        self.DEBUG_DIR = "debug"
        
        # Initialisation des sous-configurations
        self.preprocessing = self.Preprocessing()
        self.contour = self.ContourDetection()
        self.verification = self.Verification()
        self.performance = self.Performance()
        self.visualization = self.Visualization()
    
    class Preprocessing:
        """Configuration pour le prétraitement des images"""
        def __init__(self):
            self.USE_ADAPTIVE = True
            self.BLUR_KERNEL_SIZE = (5, 5)
            self.CLAHE_CLIP_LIMIT = 2.0
            self.CLAHE_GRID_SIZE = (8, 8)
            self.USE_AUTO_THRESHOLD = True
            self.ADAPTIVE_BLOCK_SIZE = 35
            self.ADAPTIVE_C = 10
            self.MORPH_KERNEL_SIZE = 5
            self.MORPH_ITERATIONS = 2
    
    class ContourDetection:
        """Configuration pour la détection des contours"""
        def __init__(self):
            self.MIN_AREA = 500  # Changed from self.config.contour.MIN_AREA
            self.MAX_AREA_RATIO = 0.3
            self.MIN_PERIMETER = 50
            self.SOLIDITY_RANGE = (0.7, 0.99)
            self.ASPECT_RATIO_RANGE = (0.25, 4.0)
            self.USE_MEAN_FILTERING = True
            self.MEAN_DEVIATION_THRESHOLD = 1.8
    
    class Verification:
        """Configuration pour la vérification des pièces"""
        def __init__(self):
            self.VALIDATION_THRESHOLD = 0.5
            self.USE_PIECE_RECOVERY = True
            self.RECOVERY_MIN_AREA_FACTOR = 0.7
    
    class Performance:
        """Configuration pour les performances"""
        def __init__(self):
            self.USE_FAST_MODE = False
            self.USE_MULTIPROCESSING = True
            self.NUM_PROCESSES = 4
    
    class Visualization:
        """Configuration pour la visualisation"""
        def __init__(self):
            self.CONTOUR_THICKNESS = 2
    
    def to_dict(self):
        """Convertit la configuration en dictionnaire plat pour compatibilité"""
        result = {
            'DEBUG': self.DEBUG,
            'DEBUG_DIR': self.DEBUG_DIR
        }
        
        # Ajouter les paramètres des sous-configurations avec préfixes
        for prefix, obj in [
            ('PREPROCESSING_', self.preprocessing),
            ('CONTOUR_', self.contour),
            ('VERIFICATION_', self.verification),
            ('PERFORMANCE_', self.performance),
            ('VISUALIZATION_', self.visualization)
        ]:
            for key in dir(obj):
                if not key.startswith('_') and not callable(getattr(obj, key)):
                    result[prefix + key] = getattr(obj, key)
        
        return result
    
    @classmethod
    def from_dict(cls, param_dict):
        """Crée une instance de configuration à partir d'un dictionnaire de paramètres"""
        config = cls()
        
        # Mise à jour des paramètres de niveau supérieur
        if 'DEBUG' in param_dict:
            config.DEBUG = param_dict['DEBUG']
        if 'DEBUG_DIR' in param_dict:
            config.DEBUG_DIR = param_dict['DEBUG_DIR']
        
        # Traitement des paramètres avec préfixes
        for key, value in param_dict.items():
            if key.startswith('PREPROCESSING_'):
                sub_key = key[14:]
                if hasattr(config.preprocessing, sub_key):
                    setattr(config.preprocessing, sub_key, value)
            elif key.startswith('CONTOUR_'):
                sub_key = key[8:]
                if hasattr(config.contour, sub_key):
                    setattr(config.contour, sub_key, value)
            # ... traitement similaire pour les autres préfixes ...
        
        return config
    
    def optimize_for_fast_processing(self):
        """Optimise les paramètres pour un traitement rapide"""
        self.preprocessing.USE_ADAPTIVE = False
        self.preprocessing.BLUR_KERNEL_SIZE = (3, 3)
        self.preprocessing.MORPH_ITERATIONS = 1
        self.preprocessing.USE_AUTO_THRESHOLD = False
        self.contour.USE_MEAN_FILTERING = False
        self.verification.VALIDATION_THRESHOLD = 0.4
        self.verification.USE_PIECE_RECOVERY = False
        self.performance.USE_FAST_MODE = True
        
    def optimize_for_image_characteristics(self, analysis):
        """
        Optimise les paramètres de configuration en fonction des caractéristiques de l'image.
        
        Args:
            analysis: Dictionnaire contenant les résultats de l'analyse de l'image
        """
        # Optimisation du prétraitement
        if 'contrast' in analysis:
            if analysis['contrast'] < 0.4:
                # Faible contraste - augmenter les paramètres CLAHE
                self.preprocessing.CLAHE_CLIP_LIMIT = 3.0
                self.preprocessing.USE_ADAPTIVE = True
            elif analysis['contrast'] > 0.7:
                # Bon contraste - paramètres plus simples
                self.preprocessing.USE_ADAPTIVE = False
        
        # Optimisation basée sur le fond
        if 'is_dark_background' in analysis and analysis['is_dark_background']:
            # Optimisations pour fond sombre (cas typique des puzzles)
            self.preprocessing.USE_AUTO_THRESHOLD = False
        
        # Optimisation basée sur la densité des bords
        if 'edge_density' in analysis:
            if analysis['edge_density'] < 0.05:
                # Image avec peu de bords - être plus permissif
                self.contour.MEAN_DEVIATION_THRESHOLD = 2.5
            elif analysis['edge_density'] > 0.15:
                # Image avec beaucoup de bords - être plus strict
                self.contour.MEAN_DEVIATION_THRESHOLD = 1.5
        
        # Appliquer les paramètres recommandés si disponibles
        if 'recommended_params' in analysis:
            rp = analysis['recommended_params']
            
            if 'MIN_CONTOUR_AREA' in rp:
                self.contour.MIN_AREA = max(500, int(rp['MIN_CONTOUR_AREA']))
                
            if 'MEAN_DEVIATION_THRESHOLD' in rp:
                self.contour.MEAN_DEVIATION_THRESHOLD = rp['MEAN_DEVIATION_THRESHOLD']
                
            if 'USE_ADAPTIVE_PREPROCESSING' in rp:
                self.preprocessing.USE_ADAPTIVE = rp['USE_ADAPTIVE_PREPROCESSING']