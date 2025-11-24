#pragma once
 
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
 
// ROS2 includes
#include "ROS2NodeComponent.h"
#include "ROS2Publisher.h"
#include "Msgs/ROS2PointCloud2.h"   // our PointCloud2 wrapper
#include "Msgs/ROS2GenericMsg.h"
 
#include "LiDAR.generated.h"
 
UCLASS()
class OFFROADSIMULATION_API ALiDAR : public AActor
{
    GENERATED_BODY()
 
public:
    ALiDAR();
    virtual void Tick(float DeltaTime) override;
 
    ////////////////////////////////////////////////////
    // LiDAR Parameters (Exposed in the Editor)
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float MaxRange;
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float HorizontalFOVStart;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float HorizontalFOVEnd;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float HorizontalResolution;
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float VerticalFOVStart;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float VerticalFOVEnd;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR")
    float VerticalResolution;
 
    ////////////////////////////////////////////////////
    // Debug Options
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    bool bDrawDebug;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    bool bShowOnlyHitPoints;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    bool bDrawOnlyPoints;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    float DebugLifetime;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    float DebugLineThickness;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    float DebugPointSize;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    float DebugCircleRadius;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LiDAR Debug")
    int32 DebugCircleSegments;
 
    UFUNCTION(BlueprintCallable, Category = "LiDAR Debug")
    void ToggleDebug();
 
    ////////////////////////////////////////////////////
    // LiDAR Data
 
    /** All ray endpoints, hits and misses */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "LiDAR Data")
    TArray<FVector> FullScanCloud;
 
    /** Only the hit (impact) points */
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "LiDAR Data")
    TArray<FVector> ImpactPointCloud;
 
    UFUNCTION(BlueprintCallable, Category = "LiDAR Data")
    const TArray<FVector>& GetImpactPointCloud() const { return ImpactPointCloud; }
 
    ////////////////////////////////////////////////////
    // ROS2 Config
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="ROS2")
    FString CloudTopic = TEXT("/lidar/points");
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="ROS2", meta=(ClampMin="1.0", ClampMax="30.0"))
    float PublishHz = 10.f;
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="ROS2")
    FString FrameId = TEXT("lidar_frame");
 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="ROS2")
    bool bSensorDataQoS = true;
 
protected:
    virtual void BeginPlay() override;
 
private:
    float RunningTime;
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* Lidar;
 
    TArray<FVector> BeamDirections;
    void PrecomputeBeamDirections();
    void PerformLiDARScan();
 
    // ROS2
    UPROPERTY(VisibleAnywhere, Category="ROS2")
    UROS2NodeComponent* ROS2Node = nullptr;
    UPROPERTY(VisibleAnywhere, Category="ROS2")
    UROS2Publisher* CloudPublisher = nullptr;
 
    UFUNCTION()
    void UpdateCloudMsg(class UROS2GenericMsg* InMsg);
};