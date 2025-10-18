#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <string.h>

// CAN hyper-period
//int h = 2;//test2HP
int h = 5; // SampleTwo

// IDset of target ECU in ascending order of periodicity
//int ECUIDs[] = {1266,1265,455,461,196,381}; //test2HP
int ECUIDs[] = {417, 451, 707, 977}; // SampleTwo.csv

// Periodicities of the IDs of target ECU in ascending order
//float ECUIDPeriodicities[] = {0.01,0.02,0.04, 0.04,0.05, 0.1}; //test2HP
float ECUIDPeriodicities[] = {0.025, 0.025, 0.05, 0.1}; // SampleTwo.csv

// length of ECUIDs
//int ECUCount = 6;//test2HP
int ECUCount = 4; // SampleTwo.csv

// minimum attack window length for successful transmissions
int minAtkWinLen = 222;

// Maximum attack window: this is to reduce space for attack window per instance
int minDlc = 1;

// Bus speed
float busSpeed = 500; // in kbps

// This is for verification
// If we want to check the analysis for a specific
int testID = 820;//461;//196;

struct Instance{
    int atkWinLen; // Length of attackwindow = total packet length of the high priority preceeding messages
    int atkWinCount; // count of high priority messages preceding the target one
    int attackable; //  if the attack window is sufficient for attacking
    int *atkWin; // List of high priority messages preceeding the target instance
};

struct Message
{
    int ID;
    float periodicity;
    int count; // no of instances per CAN hyper period
    int DLC; // Data field length in terms of byte
    float txTime;
    int atkWinLen;
    int tAtkWinLen;
    int tAtkWinCount;
    int readCount; // no. of times it is read from CAN traffic
    int *tAtkWin;
    struct Instance *instances;
};

/** *ID_set= list of structure of type ID,
n = no. of items in ID_set
IDs = list of IDs transmitted to CAN from victim
**/
void InitializeECU(struct Message **IDSet)
{
    int upper=64, lower=0, i=0,j=0;

    for(i=0;i<ECUCount;i++)
    {
        (*IDSet)[i].ID = ECUIDs[i];
        (*IDSet)[i].periodicity = ECUIDPeriodicities[i];
        (*IDSet)[i].count = ceil(h/(*IDSet)[i].periodicity);
        (*IDSet)[i].DLC = 0;
        (*IDSet)[i].atkWinLen = 0;
        (*IDSet)[i].tAtkWinLen = 0;
        (*IDSet)[i].tAtkWinCount = 0;
        (*IDSet)[i].readCount = 0;
        (*IDSet)[i].instances = (struct Instance*)calloc((*IDSet)[i].count,sizeof(struct Instance));
        for(j=0;j<(*IDSet)[i].count;j++)
        {
            (*IDSet)[i].instances[j].atkWinLen = 0;
            (*IDSet)[i].instances[j].attackable = 0;
            (*IDSet)[i].instances[j].atkWinCount = 0;
        }
    }
}

/** *This function parse the CAN trraffic from sampleOne.csv
We retrieve ID, DLC, and transmission start time
**/
int InitializeCANTraffic(struct Message **can)
{
    int row = 0, column = 0, line = 0;
    //FILE* fp = fopen("test2HP.csv", "r");//fopen("high_load_log_can_seq_1.txt","r");
    FILE* fp = fopen("SampleTwo.csv", "r");//fopen("high_load_log_can_seq_1.txt","r");
    if (!fp){
        printf("Error: Could not open 'SampleTwo.csv'.\n");
    printf("The program will now exit.\n");
    exit(1);

    // printf("Can't open file\n");
    }
        
    else
    {
        char buffer[1024];

        fgets(buffer, 1024, fp);//--> Added line

        while (fgets(buffer,3000, fp))
        {
            column = 0;
            row++;
            // Splitting the data
            char* value = strtok(buffer, ",");
            if(value!="Chn" || value!="Logging" || row>1)
            {
                line++;
                *can = (struct Message *)realloc(*can,sizeof(struct Message)*line);
                while (value) {
                    // This is for our ECU-setup and CAN log
                    if (column == 1 && line>1) {
                        (*can)[line-2].ID = (int)strtol(value, NULL, 16);//atoi(value);
                    }
                    if (column == 2 && line>1) {
                        (*can)[line-2].DLC = atoi(value);
                    }
                    if (column == 11 && line>1) {
                        (*can)[line-2].txTime = atof(value);
                    }
                    value = strtok(NULL, ",");
                    column++;
                }


                // **** ADD THIS PRINTF STATEMENT HERE ****
                // if (line > 1) { // Prevents printing for the header
                //     printf("Read Row %d: ID=0x%X, DLC=%d, Time=%f\n",
                //            row,
                //            (*can)[line-2].ID,
                //            (*can)[line-2].DLC,
                //            (*can)[line-2].txTime);
                // }
            }
        }
        fclose(fp);
        }
        line = line-2;
        *can = (struct Message *)realloc(*can,sizeof(struct Message)*line);
        return line;
}

// merge two sorted arrays
void IntMerge(int *arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temp arrays
    int L[n1], R[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp arrays back into arr[l..r
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[],
    // if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[],
    // if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// To sort an array of integers
// l and r are left and right most index of arr
void IntSort(int *arr, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;

        // Sort first and second halves
        IntSort(arr, l, m);
        IntSort(arr, m + 1, r);
        IntMerge(arr, l, m, r);
    }

}

// MErge two lists of mesages sorted by attack length
void MsgMergeByAtkWinLen(struct Message **arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temp arrays
    struct Message *L = (struct Message *)calloc(n1,sizeof(struct Message));
    struct Message *R = (struct Message *)calloc(n2,sizeof(struct Message));
    //int L[n1], R[n2];

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = (*arr)[l + i];
    for (j = 0; j < n2; j++)
        R[j] = (*arr)[m + 1 + j];

    // Merge the temp arrays back into arr[l..r
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i].atkWinLen <= R[j].atkWinLen) {
            (*arr)[k] = L[i];
            i++;
        }
        else {
            (*arr)[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[],
    // if there are any
    while (i < n1) {
        (*arr)[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[],
    // if there are any
    while (j < n2) {
        (*arr)[k] = R[j];
        j++;
        k++;
    }
    free(L);
    free(R);
}


// To sort a message list by their attack length in ascending order
void MsgSortByAtkWinLen(struct Message **candidates, int l, int r)
{
        if (l < r) {
            int m = l + (r - l) / 2;
            // Sort first and second halves
            MsgSortByAtkWinLen(candidates, l, m);
            MsgSortByAtkWinLen(candidates, m + 1, r);
            MsgMergeByAtkWinLen(candidates, l, m, r);
    }

}

int BinarySearch(int *arr, int l, int r, int x)
{
    while (l <= r) {
        int m = l + (r - l) / 2;

        // Check if x is present at mid
        if (arr[m] == x)
        {
            //printf("\n array element=%d, item to be searched=%d", arr[m],x);
            return m;
        }
        // If x greater, ignore left half
        if (arr[m] < x)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // If we reach here, then element was not present
    return -1;
}

// Returns the intersection of two arrays and b
// Update attack window of instance ins with the common messages
// void CommonMessages(int *a, int n_a, int *b, int n_b, struct Instance *ins)
// {
//     int j = 0, i=0, k=0;
//     int atkWinCount = 0;
//     int *intersection;// = (int *)calloc(j, sizeof(int));
//     if(n_a<=n_b)
//     {
//         IntSort(a, 0, n_a-1);
//         for(i=0;i<n_b;i++)
//         {
//             if(BinarySearch(a, 0, n_a-1, b[i])>=0)
//             {
//                 j++;
//                 if(j==1)
//                     intersection = (int *)calloc(j, sizeof(int));
//                 else
//                     intersection = (int *)realloc(intersection, sizeof(int)*j);
//                 intersection[j-1] = b[i];
//                 atkWinCount++;
//             }
//         }
//     }
//     else
//     {
//         IntSort(b, 0, n_b-1);
//         for(i=0;i<n_a;i++)
//         {
//             if(BinarySearch(b, 0, n_b-1, a[i])>=0)
//             {
//                 j++;
//                 if(j==1)
//                     intersection = (int *)calloc(j, sizeof(int));
//                 else
//                     intersection = (int *)realloc(intersection, sizeof(int)*j);
//                 intersection[j-1] = a[i];
//                 atkWinCount++;
//             }
//         }
//     }

//     free((*ins).atkWin);
//     (*ins).atkWinCount = atkWinCount;
//     if(atkWinCount>0)
//     {
//         (*ins).atkWin = (int *)calloc(atkWinCount, sizeof(int));
//         for(int i=0;i<atkWinCount;i++)
//         {
//             (*ins).atkWin[i] = intersection[i];
//         }
//         free(intersection);
//     }

// }

// Returns the intersection of two arrays a and b
// Update attack window of instance ins with the common messages
void CommonMessages(int *a, int n_a, int *b, int n_b, struct Instance *ins)
{
    int i = 0, j = 0, k = 0;
    int atkWinCount = 0;
    int *intersection = NULL; // Start with NULL

    if (n_a > 0 && n_b > 0) {
        if (n_a <= n_b) {
            IntSort(a, 0, n_a - 1);
            for (i = 0; i < n_b; i++) {
                if (BinarySearch(a, 0, n_a - 1, b[i]) >= 0) {
                    atkWinCount++;
                    intersection = (int *)realloc(intersection, sizeof(int) * atkWinCount);
                    intersection[atkWinCount - 1] = b[i];
                }
            }
        } else {
            IntSort(b, 0, n_b - 1);
            for (i = 0; i < n_a; i++) {
                if (BinarySearch(b, 0, n_b - 1, a[i]) >= 0) {
                    atkWinCount++;
                    intersection = (int *)realloc(intersection, sizeof(int) * atkWinCount);
                    intersection[atkWinCount - 1] = a[i];
                }
            }
        }
    }

    // Safely free the old memory and assign the new memory
    if ((*ins).atkWin != NULL) {
        free((*ins).atkWin);
    }

    (*ins).atkWin = intersection; // Assign the new list (which could be NULL if empty)
    (*ins).atkWinCount = atkWinCount;
}

void AnalyzeCANTraffic(struct Message *CANTraffic, int CANCount, struct Message **candidates)
{
    int j=0,i=0,k=0;
    float txStart = 0, txEnds = 0, nextTxStart = 0;
    float maxIdle = (minDlc*8+47)/(busSpeed*1000);
    struct Message CANPacket, candidate;
    while(j<CANCount-1)
    {
        CANPacket = CANTraffic[j];
        txStart = CANPacket.txTime;
        txEnds = ((CANPacket.DLC)*8 + 47)/(busSpeed*1000);
        nextTxStart = CANTraffic[j+1].txTime;
        for(i=0;i<ECUCount;i++)
        {
            if((CANPacket.ID > (*candidates)[i].ID) || ((nextTxStart - (txStart + txEnds))>maxIdle && (CANPacket.ID != (*candidates)[i].ID))) // If CAN packet is of lower priority
            {
                if((*candidates)[i].tAtkWinLen>0)
                {
                    free((*candidates)[i].tAtkWin);
                    (*candidates)[i].tAtkWinLen = 0;
                    (*candidates)[i].tAtkWinCount = 0;
                }
            }
            else if((CANPacket.ID < (*candidates)[i].ID)) // If CAN packet belongs to attack window
            {
                (*candidates)[i].tAtkWinCount = (*candidates)[i].tAtkWinCount + 1;
                (*candidates)[i].tAtkWinLen = (*candidates)[i].tAtkWinLen + (CANPacket.DLC)*8 + 47;
                if((*candidates)[i].tAtkWinCount == 1)
                    (*candidates)[i].tAtkWin = (int *)calloc((*candidates)[i].tAtkWinCount,sizeof(int));
                else
                    (*candidates)[i].tAtkWin = (int *)realloc((*candidates)[i].tAtkWin,sizeof(int)*(*candidates)[i].tAtkWinCount);
                (*candidates)[i].tAtkWin[(*candidates)[i].tAtkWinCount-1] = CANPacket.ID;
            }
            else
            {
                if((*candidates)[i].readCount>=(*candidates)[i].count) // 2nd hyper period onwards
                {

                    (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinLen = (int)fmin((*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinLen, (*candidates)[i].tAtkWinLen);
                    if((*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinLen == 0)
                    {
                        (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount = 0;
                        (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWin =
                                                            (int *)calloc((*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount,sizeof(int));
                    }
                    else{
                    CommonMessages((*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWin,
                                   (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount,
                                   (*candidates)[i].tAtkWin,
                                   (*candidates)[i].tAtkWinCount,
                                   &(*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count]);
                    }
                }
                else // 1st hyper period
                {
                    (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinLen = (*candidates)[i].tAtkWinLen;
                    (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount = (*candidates)[i].tAtkWinCount;
                    (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWin =
                                                            (int *)calloc((*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount,sizeof(int));
                    for(k=0;k<(*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWinCount;k++)
                    {
                        (*candidates)[i].instances[(*candidates)[i].readCount%(*candidates)[i].count].atkWin[k] = (*candidates)[i].tAtkWin[k];
                    }
                }

                if((*candidates)[i].tAtkWinLen>0)
                {
                    free((*candidates)[i].tAtkWin);
                    (*candidates)[i].tAtkWinLen = 0;
                    (*candidates)[i].tAtkWinCount = 0;
                }
                (*candidates)[i].readCount=(*candidates)[i].readCount+1;
            }
        }
        j++;
    }
}


int main()
{
    int i=0, sum=0, j=0, k=0, CANCount = 0;
    float smallestPeriod = 0;

    srand(time(0));

    struct Message *CANTraffic = (struct Message *)calloc(CANCount+1,sizeof(struct Message));
    struct Message *candidates = (struct Message *)calloc(ECUCount,sizeof(struct Message));
    struct Message *sortecCandidates = (struct Message *)calloc(1,sizeof(struct Message));

    // CANCount = InitializeCANTraffic(&CANTraffic);
    // InitializeECU(&candidates);
    // AnalyzeCANTraffic(CANTraffic,CANCount,&candidates);

    // **** ADD DEBUG PRINTS AROUND EACH MAJOR FUNCTION CALL ****
    printf("DEBUG: Starting analysis...\n");
    CANCount = InitializeCANTraffic(&CANTraffic);
    printf("DEBUG: Finished InitializeCANTraffic. Messages read: %d\n", CANCount);

    InitializeECU(&candidates);
    printf("DEBUG: Finished InitializeECU.\n");

    AnalyzeCANTraffic(CANTraffic,CANCount,&candidates);
    printf("DEBUG: Finished AnalyzeCANTraffic.\n"); // <-- I suspect it will crash before this line.


    //This step can only be achieved if candidates is sorted in ascending order wrt periodicity
    // By default we have taken the input in sorted order
    smallestPeriod = candidates[0].periodicity;
    k = 0;
    printf("\nThe attackablility of %d is:",testID);
    for(i=0;i<ECUCount;i++)
    {
        sum = 0;
        for(j=0;j<candidates[i].count;j++)
        {
           if(candidates[i].instances[j].atkWinLen>=minAtkWinLen)
                candidates[i].instances[j].attackable = 1;
            else
                candidates[i].instances[j].attackable = 0;
            sum += candidates[i].instances[j].atkWinLen;
            if(candidates[i].ID == testID)
                printf("%d  ",candidates[i].instances[j].attackable);
        }
        candidates[i].atkWinLen = sum/candidates[i].count;
        if(candidates[i].atkWinLen>=minAtkWinLen)
        {
            k++;
            sortecCandidates = (struct Message *)realloc(sortecCandidates,sizeof(struct Message)*k);
            sortecCandidates[k-1] = candidates[i];
        }

    }
    // We have to sort the candidate list in ascending order wrt to periodicity
    // But we have already kept them in sorted order only
    MsgSortByAtkWinLen(&sortecCandidates,0,k-1);
    printf("\nThe target ID is:=%d\n",sortecCandidates[k-1].ID);

    free(sortecCandidates);
    free(candidates);
    free(CANTraffic);
    return 0;
}
