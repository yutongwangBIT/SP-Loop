#include "camodocal/camera_models/CameraFactory.h"

#include <boost/algorithm/string.hpp>

#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/PinholeFullCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include "ceres/ceres.h"

namespace camodocal
{

boost::shared_ptr< CameraFactory > CameraFactory::m_instance;

CameraFactory::CameraFactory( ) {}

boost::shared_ptr< CameraFactory >
CameraFactory::instance( void )
{
    if ( m_instance.get( ) == 0 )
    {
        m_instance.reset( new CameraFactory );
    }

    return m_instance;
}

CameraPtr
CameraFactory::generateCamera( Camera::ModelType modelType, const std::string& cameraName, cv::Size imageSize ) const
{
    switch ( modelType )
    {
        case Camera::KANNALA_BRANDT:
        {
            EquidistantCameraPtr camera( new EquidistantCamera );

            EquidistantCamera::Parameters params = camera->getParameters( );
            params.cameraName( )                 = cameraName;
            params.imageWidth( )                 = imageSize.width;
            params.imageHeight( )                = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera( new PinholeCamera );

            PinholeCamera::Parameters params = camera->getParameters( );
            params.cameraName( )             = cameraName;
            params.imageWidth( )             = imageSize.width;
            params.imageHeight( )            = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE_FULL:
        {
            PinholeFullCameraPtr camera( new PinholeFullCamera );

            PinholeFullCamera::Parameters params = camera->getParameters( );
            params.cameraName( )                 = cameraName;
            params.imageWidth( )                 = imageSize.width;
            params.imageHeight( )                = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::SCARAMUZZA:
        {
            OCAMCameraPtr camera( new OCAMCamera );

            OCAMCamera::Parameters params = camera->getParameters( );
            params.cameraName( )          = cameraName;
            params.imageWidth( )          = imageSize.width;
            params.imageHeight( )         = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
        case Camera::MEI:
        default:
        {
            CataCameraPtr camera( new CataCamera );

            CataCamera::Parameters params = camera->getParameters( );
            params.cameraName( )          = cameraName;
            params.imageWidth( )          = imageSize.width;
            params.imageHeight( )         = imageSize.height;
            camera->setParameters( params );
            return camera;
        }
    }
}

CameraPtr
CameraFactory::generateCameraFromYamlFile( const std::string& filename )
{
    //std::cout<<"filename:"<<filename<<std::endl;
    cv::FileStorage fs( filename, cv::FileStorage::READ );
    //std::cout<<"FileStorage want open"<<std::endl;
    if ( !fs.isOpened( ) )
    {
        //std::cout<<"CameraFactory::generateCameraFromYamlFile file not opened"<<std::endl;
        return CameraPtr( );
    }
   /* else{
        std::cout<<"CameraFactory::generateCameraFromYamlFile file opened"<<std::endl;
    }*/

    Camera::ModelType modelType = Camera::MEI;
    if ( !fs["model_type"].isNone( ) )
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;
        std::cout<<"sModelType:"<<sModelType<<std::endl;

        if ( boost::iequals( sModelType, "kannala_brandt" ) )
        {
            modelType = Camera::KANNALA_BRANDT;
        }
        else if ( boost::iequals( sModelType, "mei" ) )
        {
            modelType = Camera::MEI;
        }
        else if ( boost::iequals( sModelType, "scaramuzza" ) )
        {
            modelType = Camera::SCARAMUZZA;
        }
        else if ( boost::iequals( sModelType, "pinhole" ) )
        {
            modelType = Camera::PINHOLE;
        }
        else if ( boost::iequals( sModelType, "PINHOLE_FULL" ) )
        {
            modelType = Camera::PINHOLE_FULL;
        }
        else
        {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
            return CameraPtr( );
        }
    }
   /* else{
        std::cerr << "# ERROR: model_type is none: "  << std::endl;
    }*/

    switch ( modelType )
    {
        case Camera::KANNALA_BRANDT:
        {
            EquidistantCameraPtr camera( new EquidistantCamera );

            EquidistantCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera( new PinholeCamera );

            PinholeCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::PINHOLE_FULL:
        {
            PinholeFullCameraPtr camera( new PinholeFullCamera );

            PinholeFullCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::SCARAMUZZA:
        {
            OCAMCameraPtr camera( new OCAMCamera );

            OCAMCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            return camera;
        }
        case Camera::MEI:
        default:
        {
            //std::cout<<"IT IS MEI"<<std::endl;
            CataCameraPtr camera( new CataCamera );

            CataCamera::Parameters params = camera->getParameters( );
            params.readFromYamlFile( filename );
            camera->setParameters( params );
            //std::cout<<"setParameters"<<std::endl;
            return camera;
        }
    }

    return CameraPtr( );
}
}
