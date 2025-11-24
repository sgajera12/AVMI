#include "Sensors/LiDAR.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Kismet/GameplayStatics.h"
 
// ROS2
#include "Msgs/ROS2PointCloud2.h"
#include "Msgs/ROS2Time.h"
#include "rclcUtilities.h"
 
static inline FROSTime MakeROSTimeNow()
{
    const FDateTime NowUTC = FDateTime::UtcNow();
    const FTimespan SinceEpoch = NowUTC - FDateTime(1970,1,1);
    FROSTime T;
    const int64 Secs = (int64)SinceEpoch.GetTotalSeconds();
    const int64 RemaTicks = SinceEpoch.GetTicks() - Secs * ETimespan::TicksPerSecond;
    T.Sec = (int)Secs;
    T.Nanosec = (unsigned int)(RemaTicks * 100); // 100ns -> ns
    return T;
}
 
ALiDAR::ALiDAR()
{
    PrimaryActorTick.bCanEverTick = true;
 
    Lidar = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("LidarMesh"));
    RootComponent = Lidar;
 
    MaxRange = 1000.0f;
    HorizontalFOVStart = -179.f;
    HorizontalFOVEnd = 180.f;
    HorizontalResolution = 1.f;
    VerticalFOVStart = -20.f;
    VerticalFOVEnd = 20.f;
    VerticalResolution = 1.f;
 
    bDrawDebug = true;
    bShowOnlyHitPoints = false;
    bDrawOnlyPoints = true;
    DebugLifetime = 0.05f;
    DebugLineThickness = 1.f;
    DebugPointSize = 5.f;
    DebugCircleRadius = 50.f;
    DebugCircleSegments = 32;
 
    RunningTime = 0.f;
}
 
void ALiDAR::BeginPlay()
{
    Super::BeginPlay();
    PrecomputeBeamDirections();
 
    // Attach to vehicle if available
    if (APawn* Vehicle = UGameplayStatics::GetPlayerPawn(this, 0))
    {
        AttachToActor(Vehicle, FAttachmentTransformRules::KeepRelativeTransform);
        SetActorRelativeRotation(FRotator::ZeroRotator);
    }
 
    // ROS2 setup
    ROS2Node = NewObject<UROS2NodeComponent>(this, UROS2NodeComponent::StaticClass(), TEXT("ROS2Node"));
    if (ROS2Node)
    {
        ROS2Node->RegisterComponent();
        ROS2Node->Init();
 
        const UROS2QoS QoS = bSensorDataQoS ? UROS2QoS::SensorData : UROS2QoS::Default;
 
        ROS2_CREATE_LOOP_PUBLISHER_WITH_QOS(
            ROS2Node,
            this,
            CloudTopic,
            UROS2Publisher::StaticClass(),
            UROS2PointCloud2Msg::StaticClass(),
            PublishHz,
&ALiDAR::UpdateCloudMsg,
            QoS,
            CloudPublisher
        );
 
        UE_LOG(LogTemp, Log, TEXT("LiDAR publishing %s @ %.1f Hz"), *CloudTopic, PublishHz);
    }
}
 
void ALiDAR::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    RunningTime += DeltaTime;
 
    FullScanCloud.Empty();
    ImpactPointCloud.Empty();
    PerformLiDARScan();
 
    if (bDrawDebug)
    {
        DrawDebugCircle(GetWorld(), GetActorLocation(), DebugCircleRadius, DebugCircleSegments,
                        FColor::Blue, false, DebugLifetime, 0,
                        DebugLineThickness, FVector(0,0,1), FVector(1,0,0), false);
    }
}
 
void ALiDAR::PrecomputeBeamDirections()
{
    BeamDirections.Empty();
    for (float Yaw = HorizontalFOVStart; Yaw <= HorizontalFOVEnd; Yaw += HorizontalResolution)
    {
        for (float Pitch = VerticalFOVStart; Pitch <= VerticalFOVEnd; Pitch += VerticalResolution)
        {
            FRotator R(Pitch, Yaw, 0.f);
            BeamDirections.Add(R.Vector());
        }
    }
}
 
void ALiDAR::PerformLiDARScan()
{
    const FVector SensorLoc = GetActorLocation();
    const FRotator SensorRot = GetActorRotation();
 
    for (const FVector& LocalDir : BeamDirections)
    {
        const FVector WorldDir = SensorRot.RotateVector(LocalDir);
        const FVector Start = SensorLoc;
        const FVector End = Start + WorldDir * MaxRange;
 
        FHitResult Hit;
        FCollisionQueryParams Params;
        Params.AddIgnoredActor(this);
 
        bool bHit = GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility, Params);
 
        if (bHit)
        {
            FullScanCloud.Add(Hit.ImpactPoint);
            ImpactPointCloud.Add(Hit.ImpactPoint);
        }
        else
        {
            FullScanCloud.Add(End);
        }
 
        if (bDrawDebug)
        {
            if (bHit)
                DrawDebugPoint(GetWorld(), Hit.ImpactPoint, DebugPointSize, FColor::Green, false, DebugLifetime);
            else if (!bShowOnlyHitPoints && !bDrawOnlyPoints)
                DrawDebugLine(GetWorld(), Start, End, FColor::Blue, false, DebugLifetime, 0, DebugLineThickness);
        }
    }
}
 
void ALiDAR::ToggleDebug()
{
    bDrawDebug = !bDrawDebug;
    if (GEngine)
    {
        FString Status = bDrawDebug ? TEXT("enabled") : TEXT("disabled");
        GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Yellow,
            FString::Printf(TEXT("LiDAR Debug %s"), *Status));
    }
}
 
void ALiDAR::UpdateCloudMsg(UROS2GenericMsg* InMsg)
{
    if (!InMsg) return;
 
    const int32 N = ImpactPointCloud.Num();
    if (N <= 0) return;
 
    FROSPointCloud2 Cloud;
    Cloud.Header.Stamp   = MakeROSTimeNow();
    Cloud.Header.FrameId = FrameId;
 
    Cloud.Height = 1;
    Cloud.Width  = N;
    Cloud.bIsBigendian = false;
    Cloud.bIsDense = true;
 
    Cloud.Fields.SetNum(3);
    Cloud.Fields[0].Name     = TEXT("x");
    Cloud.Fields[0].Offset   = 0;
    Cloud.Fields[0].Datatype = 7; // FLOAT32
    Cloud.Fields[0].Count    = 1;
 
    Cloud.Fields[1].Name     = TEXT("y");
    Cloud.Fields[1].Offset   = 4;
    Cloud.Fields[1].Datatype = 7;
    Cloud.Fields[1].Count    = 1;
 
    Cloud.Fields[2].Name     = TEXT("z");
    Cloud.Fields[2].Offset   = 8;
    Cloud.Fields[2].Datatype = 7;
    Cloud.Fields[2].Count    = 1;
 
    const uint32 PointStep = 12;
    Cloud.PointStep = PointStep;
    Cloud.RowStep   = PointStep * Cloud.Width;
    Cloud.Data.SetNumUninitialized(Cloud.RowStep);
 
    uint8* dst = Cloud.Data.GetData();
    for (int32 i = 0; i < N; ++i)
    {
        const FVector& Pcm = ImpactPointCloud[i];
        const float xm =  Pcm.X / 100.0f;
        const float ym = -Pcm.Y / 100.0f; // flip Y
        const float zm =  Pcm.Z / 100.0f;
 
        FMemory::Memcpy(dst + i*PointStep + 0, &xm, sizeof(float));
        FMemory::Memcpy(dst + i*PointStep + 4, &ym, sizeof(float));
        FMemory::Memcpy(dst + i*PointStep + 8, &zm, sizeof(float));
    }
 
    if (UROS2PointCloud2Msg* Out = Cast<UROS2PointCloud2Msg>(InMsg))
    {
        Out->SetMsg(Cloud);
    }
}