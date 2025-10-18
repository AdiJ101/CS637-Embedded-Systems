#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include <string.h>

// CAN hyper-period
int h = 5; // SampleTwo

// IDset of target ECU in ascending order of periodicity
int ECUIDs[] = {417, 451, 707, 977}; // SampleTwo.csv

// Periodicities of the IDs of target ECU in ascending order
float ECUIDPeriodicities[] = {0.025, 0.025, 0.05, 0.1}; // SampleTwo.csv

// length of ECUIDs
int ECUCount = 4; // SampleTwo.csv

// minimum attack window length for successful transmissions
int minAtkWinLen = 222;

// Bus speed
float busSpeed = 500; // in kbps

struct Instance{
    int atkWinLen;
    int atkWinCount;
    int attackable;
    int *atkWin;
};

struct Message
{
    int ID;
    float periodicity;
    int count;
    int DLC;
    float txTime;
    int atkWinLen;
    int tAtkWinLen;
    int tAtkWinCount;
    int readCount;
    int *tAtkWin;
    struct Instance *instances;
};

void InitializeECU(struct Message **IDSet)
{
    int i=0,j=0;

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
        (*IDSet)[i].tAtkWin = NULL;
        (*IDSet)[i].instances = (struct Instance*)calloc((*IDSet)[i].count,sizeof(struct Instance));
        for(j=0;j<(*IDSet)[i].count;j++)
        {
            (*IDSet)[i].instances[j].atkWinLen = 0;
            (*IDSet)[i].instances[j].attackable = 0;
            (*IDSet)[i].instances[j].atkWinCount = 0;
            (*IDSet)[i].instances[j].atkWin = NULL;
        }
    }
}

int InitializeCANTraffic(struct Message **can)
{
    int row = 0, column = 0, line = 0;
    FILE* fp = fopen("SampleTwo.csv", "r");
    if (!fp){
        printf("Error: Could not open 'SampleTwo.csv'.\n");
        printf("Please make sure the file is in the same directory as the executable.\n");
        exit(1);
    }

    char buffer[3000];
    fgets(buffer, 3000, fp);

    while (fgets(buffer, 3000, fp))
    {
        column = 0;
        row++;
        char* value = strtok(buffer, ",");

        line++;
        *can = (struct Message *)realloc(*can, sizeof(struct Message) * line);

        while (value) {
            if (column == 1) {
                (*can)[line-1].ID = (int)strtol(value, NULL, 16);
            }
            if (column == 2) {
                (*can)[line-1].DLC = atoi(value);
            }
            if (column == 11) {
                (*can)[line-1].txTime = atof(value);
            }
            value = strtok(NULL, ",");
            column++;
        }
    }
    fclose(fp);
    return line;
}

void IntMerge(int *arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];
    for (i = 0; i < n1; i++) L[i] = arr[l + i];
    for (j = 0; j < n2; j++) R[j] = arr[m + 1 + j];
    i = 0; j = 0; k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void IntSort(int *arr, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
        IntSort(arr, l, m);
        IntSort(arr, m + 1, r);
        IntMerge(arr, l, m, r);
    }
}

void MsgMergeByAtkWinLen(struct Message **arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;
    struct Message *L = (struct Message *)calloc(n1,sizeof(struct Message));
    struct Message *R = (struct Message *)calloc(n2,sizeof(struct Message));
    for (i = 0; i < n1; i++) L[i] = (*arr)[l + i];
    for (j = 0; j < n2; j++) R[j] = (*arr)[m + 1 + j];
    i = 0; j = 0; k = l;
    while (i < n1 && j < n2) {
        if (L[i].atkWinLen <= R[j].atkWinLen) (*arr)[k++] = L[i++];
        else (*arr)[k++] = R[j++];
    }
    while (i < n1) (*arr)[k++] = L[i++];
    while (j < n2) (*arr)[k++] = R[j++];
    free(L);
    free(R);
}

void MsgSortByAtkWinLen(struct Message **candidates, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
        MsgSortByAtkWinLen(candidates, l, m);
        MsgSortByAtkWinLen(candidates, m + 1, r);
        MsgMergeByAtkWinLen(candidates, l, m, r);
    }
}

int BinarySearch(int *arr, int l, int r, int x)
{
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == x) return m;
        if (arr[m] < x) l = m + 1;
        else r = m - 1;
    }
    return -1;
}

void CommonMessages(int *a, int n_a, int *b, int n_b, struct Instance *ins)
{
    int i = 0;
    int atkWinCount = 0;
    int *intersection = NULL;

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

    if ((*ins).atkWin != NULL) {
        free((*ins).atkWin);
    }

    (*ins).atkWin = intersection;
    (*ins).atkWinCount = atkWinCount;
}

void AnalyzeCANTraffic(struct Message *CANTraffic, int CANCount, struct Message **candidates)
{
    int j=0,i=0,k=0;
    for(j=0; j < CANCount-1; j++)
    {
        struct Message CANPacket = CANTraffic[j];
        for(i=0; i<ECUCount; i++)
        {
            if(CANPacket.ID > (*candidates)[i].ID)
            {
                if((*candidates)[i].tAtkWinLen > 0)
                {
                    free((*candidates)[i].tAtkWin);
                    (*candidates)[i].tAtkWin = NULL;
                    (*candidates)[i].tAtkWinLen = 0;
                    (*candidates)[i].tAtkWinCount = 0;
                }
            }
            else if(CANPacket.ID < (*candidates)[i].ID)
            {
                (*candidates)[i].tAtkWinCount++;
                (*candidates)[i].tAtkWinLen += (CANPacket.DLC)*8 + 47;
                (*candidates)[i].tAtkWin = (int *)realloc((*candidates)[i].tAtkWin, sizeof(int)*(*candidates)[i].tAtkWinCount);
                (*candidates)[i].tAtkWin[(*candidates)[i].tAtkWinCount-1] = CANPacket.ID;
            }
            else
            {
                int instance_idx = (*candidates)[i].readCount % (*candidates)[i].count;
                if((*candidates)[i].readCount >= (*candidates)[i].count)
                {
                    (*candidates)[i].instances[instance_idx].atkWinLen = (int)fmin((*candidates)[i].instances[instance_idx].atkWinLen, (*candidates)[i].tAtkWinLen);
                    CommonMessages((*candidates)[i].instances[instance_idx].atkWin,
                                   (*candidates)[i].instances[instance_idx].atkWinCount,
                                   (*candidates)[i].tAtkWin,
                                   (*candidates)[i].tAtkWinCount,
                                   &(*candidates)[i].instances[instance_idx]);
                }
                else
                {
                    (*candidates)[i].instances[instance_idx].atkWinLen = (*candidates)[i].tAtkWinLen;
                    (*candidates)[i].instances[instance_idx].atkWinCount = (*candidates)[i].tAtkWinCount;
                    (*candidates)[i].instances[instance_idx].atkWin = (int *)calloc((*candidates)[i].tAtkWinCount,sizeof(int));
                    for(k=0; k < (*candidates)[i].tAtkWinCount; k++)
                    {
                        (*candidates)[i].instances[instance_idx].atkWin[k] = (*candidates)[i].tAtkWin[k];
                    }
                }

                if((*candidates)[i].tAtkWinLen > 0)
                {
                    free((*candidates)[i].tAtkWin);
                    (*candidates)[i].tAtkWin = NULL;
                    (*candidates)[i].tAtkWinLen = 0;
                    (*candidates)[i].tAtkWinCount = 0;
                }
                (*candidates)[i].readCount++;
            }
        }
    }
}

int main()
{
    int i=0, sum=0, j=0, k=0, CANCount = 0;

    struct Message *CANTraffic = NULL;
    struct Message *candidates = (struct Message *)calloc(ECUCount,sizeof(struct Message));
    struct Message *sortecCandidates = NULL;

    printf("Starting CAN traffic analysis...\n");
    CANCount = InitializeCANTraffic(&CANTraffic);
    InitializeECU(&candidates);
    AnalyzeCANTraffic(CANTraffic,CANCount,&candidates);
    printf("Analysis complete. Finding most vulnerable target...\n");

    k = 0;
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
        }

        if (candidates[i].count > 0) {
            candidates[i].atkWinLen = sum / candidates[i].count;
        } else {
            candidates[i].atkWinLen = 0;
        }

        if(candidates[i].atkWinLen>=minAtkWinLen)
        {
            k++;
            sortecCandidates = (struct Message *)realloc(sortecCandidates,sizeof(struct Message)*k);
            sortecCandidates[k-1] = candidates[i];
        }
    }

    if (k > 0) {
        MsgSortByAtkWinLen(&sortecCandidates,0,k-1);
        
        // --- NEW CODE: Generate and print the vulnerability profile ---
        
        // 1. Get the ID of the best target
        int best_target_id = sortecCandidates[k-1].ID;
        struct Message* finalTarget = NULL;

        // 2. Find the original message struct to get all instance details
        for(i=0; i<ECUCount; i++) {
            if (candidates[i].ID == best_target_id) {
                finalTarget = &candidates[i];
                break;
            }
        }
        
        // 3. Calculate the number of attackable instances
        int attackable_count = 0;
        if (finalTarget != NULL) {
            for (j = 0; j < finalTarget->count; j++) {
                if (finalTarget->instances[j].attackable == 1) {
                    attackable_count++;
                }
            }
            
            float attackable_percentage = 0.0;
            if (finalTarget->count > 0) {
                attackable_percentage = ((float)attackable_count / finalTarget->count) * 100.0;
            }

            // 4. Print the final, detailed profile
            printf("\n--- Vulnerability Profile for Target ID: %d (0x%X) ---\n", finalTarget->ID, finalTarget->ID);
            printf("Periodicity: %.3f s\n", finalTarget->periodicity);
            printf("Instances per Hyper-period: %d\n", finalTarget->count);
            printf("Attackable Instances: %d (%.2f%%)\n", attackable_count, attackable_percentage);
            printf("Average Attack Window Length: %d bits\n", finalTarget->atkWinLen);
            printf("---------------------------------------------------\n");
        }

    } else {
        printf("\nNo suitable target ID found with the given criteria.\n");
    }

    // Free all allocated memory to prevent leaks
    free(CANTraffic);
    if (sortecCandidates != NULL) {
        free(sortecCandidates);
    }
    for(i=0; i<ECUCount; i++) {
        for(j=0; j<candidates[i].count; j++) {
            if (candidates[i].instances[j].atkWin != NULL) {
                free(candidates[i].instances[j].atkWin);
            }
        }
        free(candidates[i].instances);
    }
    free(candidates);

    return 0;
}